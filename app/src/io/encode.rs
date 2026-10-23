//! NVENC encoder fed directly from CUDA hardware frames (zero copy from fb_native kernels).

use super::probe::*;
use super::*;
use crate::native::YuvFormat;
use crate::native::YuvFrame;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoCodec {
    H264,
    Hevc,
    Av1,
}

impl VideoCodec {
    pub fn parse(s: &str) -> Option<VideoCodec> {
        match s.trim().to_ascii_lowercase().as_str() {
            "h264" | "avc" | "h.264" => Some(VideoCodec::H264),
            "hevc" | "h265" | "h.265" => Some(VideoCodec::Hevc),
            "av1" => Some(VideoCodec::Av1),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            VideoCodec::H264 => "h264",
            VideoCodec::Hevc => "hevc",
            VideoCodec::Av1 => "av1",
        }
    }
    pub fn encoder_name(self) -> &'static str {
        match self {
            VideoCodec::H264 => "h264_nvenc",
            VideoCodec::Hevc => "hevc_nvenc",
            VideoCodec::Av1 => "av1_nvenc",
        }
    }
    /// Largest frame dimension NVENC accepts for this codec (Ampere / Ada).
    pub fn max_dim(self) -> i32 {
        match self {
            VideoCodec::H264 => 4096,
            VideoCodec::Hevc | VideoCodec::Av1 => 8192,
        }
    }
}

/// Does an NVIDIA GPU with this compute capability have an NVENC engine for `codec`?
/// AV1 encoding needs Ada Lovelace (compute capability 8.9) or newer.
pub fn nvenc_supports(codec: VideoCodec, cc_major: i32, cc_minor: i32) -> bool {
    match codec {
        VideoCodec::H264 | VideoCodec::Hevc => cc_major >= 5,
        VideoCodec::Av1 => (cc_major, cc_minor) >= (8, 9),
    }
}

#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub codec: VideoCodec,
    pub cq: u32,
    /// NVENC preset (`p1`..`p7`).
    pub preset: String,
    pub width: i32,
    pub height: i32,
    /// Output frame rate (also gives the time base `1/fps`).
    pub fps: AVRational,
    /// P010 (10-bit) instead of NV12.
    pub ten_bit: bool,
    pub color: ColorSpec,
    pub sar: AVRational,
    pub global_header: bool,
    /// (cc_major, cc_minor) of the GPU, to reject unsupported codecs early.
    pub gpu_cc: Option<(i32, i32)>,
}

pub struct VideoEncoder {
    ctx: *mut ff::AVCodecContext,
    frames: BufRef,
    pkt: Packet,
    pub cfg: EncoderConfig,
    pub time_base: AVRational,
    pub format: YuvFormat,
    pub flushed: bool,
}

unsafe impl Send for VideoEncoder {}

fn set_opt(priv_data: *mut std::ffi::c_void, name: &str, val: &str) {
    let (n, v) = (CString::new(name).unwrap(), CString::new(val).unwrap());
    let ret = unsafe { ff::av_opt_set(priv_data, n.as_ptr(), v.as_ptr(), 0) };
    if ret < 0 {
        tracing::warn!("encoder option {name}={val} rejected: {}", av_strerror(ret));
    }
}

impl VideoEncoder {
    pub fn open(cfg: &EncoderConfig, device: &CudaDevice) -> Result<VideoEncoder> {
        let codec = cfg.codec;
        if let Some((maj, min)) = cfg.gpu_cc {
            if !nvenc_supports(codec, maj, min) {
                return Err(IoError::Unsupported(format!(
                    "this GPU (compute capability {maj}.{min}) has no {} hardware encoder. \
                     AV1 encoding needs an RTX 40-series (Ada) or newer; use --codec hevc or --codec h264",
                    codec.name()
                )));
            }
        }
        if cfg.width > codec.max_dim() || cfg.height > codec.max_dim() {
            return Err(IoError::Unsupported(format!(
                "output size {}x{} exceeds the {} NVENC limit of {}; use --codec hevc or a smaller upscale factor",
                cfg.width,
                cfg.height,
                codec.name(),
                codec.max_dim()
            )));
        }
        let name = cstring(codec.encoder_name())?;
        let enc = unsafe { ff::avcodec_find_encoder_by_name(name.as_ptr()) };
        if enc.is_null() {
            return Err(IoError::Unsupported(format!(
                "this FFmpeg build has no '{}' encoder",
                codec.encoder_name()
            )));
        }
        let sw_format = if cfg.ten_bit && codec != VideoCodec::H264 { ff::AV_PIX_FMT_P010LE } else { ff::AV_PIX_FMT_NV12 };
        let format = if sw_format == ff::AV_PIX_FMT_P010LE { YuvFormat::P010 } else { YuvFormat::Nv12 };
        unsafe {
            // hardware frames pool (CUDA device memory, NV12/P010)
            let r = ff::av_hwframe_ctx_alloc(device.as_ptr());
            if r.is_null() {
                return Err(IoError::Other("av_hwframe_ctx_alloc failed".into()));
            }
            let frames = BufRef::from_raw(r);
            let fc = (*r).data as *mut ff::AVHWFramesContext;
            (*fc).format = ff::AV_PIX_FMT_CUDA;
            (*fc).sw_format = sw_format;
            (*fc).width = cfg.width;
            (*fc).height = cfg.height;
            (*fc).initial_pool_size = 0;
            ffcheck(ff::av_hwframe_ctx_init(r), "av_hwframe_ctx_init(encoder)")?;

            let mut ctx = ff::avcodec_alloc_context3(enc);
            if ctx.is_null() {
                return Err(IoError::Other("avcodec_alloc_context3 failed".into()));
            }
            let tb = AVRational { num: cfg.fps.den, den: cfg.fps.num };
            (*ctx).width = cfg.width;
            (*ctx).height = cfg.height;
            (*ctx).time_base = tb;
            (*ctx).framerate = cfg.fps;
            (*ctx).pix_fmt = ff::AV_PIX_FMT_CUDA;
            (*ctx).hw_frames_ctx = frames.new_ref();
            (*ctx).sample_aspect_ratio = cfg.sar;
            (*ctx).bit_rate = 0;
            let fps_f = q2d(cfg.fps);
            (*ctx).gop_size = ((fps_f * 2.0).round() as i32).clamp(12, 600);
            if codec != VideoCodec::Av1 {
                (*ctx).max_b_frames = 2;
            }
            (*ctx).color_range = cfg.color.tags.range as _;
            (*ctx).colorspace = cfg.color.tags.space as _;
            (*ctx).color_primaries = cfg.color.tags.primaries as _;
            (*ctx).color_trc = cfg.color.tags.trc as _;
            if cfg.global_header {
                (*ctx).flags |= ff::AV_CODEC_FLAG_GLOBAL_HEADER as i32;
            }
            let pd = (*ctx).priv_data;
            set_opt(pd, "preset", &cfg.preset);
            set_opt(pd, "tune", "hq");
            set_opt(pd, "rc", "vbr");
            set_opt(pd, "cq", &cfg.cq.to_string());
            set_opt(pd, "spatial-aq", "1");
            match (codec, sw_format == ff::AV_PIX_FMT_P010LE) {
                (VideoCodec::Hevc, true) => set_opt(pd, "profile", "main10"),
                (VideoCodec::Hevc, false) => set_opt(pd, "profile", "main"),
                (VideoCodec::H264, _) => set_opt(pd, "profile", "high"),
                (VideoCodec::Av1, _) => {}
            }
            let ret = ff::avcodec_open2(ctx, enc, ptr::null_mut());
            if ret < 0 {
                ff::avcodec_free_context(&mut ctx);
                let hint = match codec {
                    VideoCodec::Av1 => " (AV1 NVENC needs an RTX 40-series GPU or newer)",
                    _ => " (is the NVIDIA driver recent enough for NVENC API 13? try updating it)",
                };
                return Err(IoError::Unsupported(format!(
                    "cannot open {}: {}{}",
                    codec.encoder_name(),
                    av_strerror(ret),
                    hint
                )));
            }
            Ok(VideoEncoder {
                ctx,
                frames,
                pkt: Packet::alloc()?,
                cfg: cfg.clone(),
                time_base: tb,
                format,
                flushed: false,
            })
        }
    }

    pub fn codec_ctx(&self) -> *mut ff::AVCodecContext {
        self.ctx
    }

    /// A fresh frame from the encoder's CUDA pool plus a view of its planes for kernels.
    pub fn alloc_frame(&self) -> Result<(Frame, YuvFrame)> {
        let f = Frame::alloc()?;
        unsafe {
            ffcheck(ff::av_hwframe_get_buffer(self.frames.as_ptr(), f.as_ptr(), 0), "av_hwframe_get_buffer(encoder)")?;
            let p = f.as_ptr();
            let yuv = YuvFrame {
                luma: (*p).data[0] as u64,
                luma_pitch: (*p).linesize[0] as usize,
                chroma: (*p).data[1] as u64,
                chroma_pitch: (*p).linesize[1] as usize,
                width: self.cfg.width,
                height: self.cfg.height,
                format: self.format,
            };
            Ok((f, yuv))
        }
    }

    /// Send a frame (`None` = flush) and hand every produced packet (timestamps in
    /// `self.time_base`) to `sink`.
    pub fn encode(
        &mut self,
        frame: Option<(&Frame, i64)>,
        sink: &mut dyn FnMut(&mut Packet) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        unsafe {
            let fp = match frame {
                Some((f, pts)) => {
                    let p = f.as_ptr();
                    (*p).pts = pts;
                    (*p).pict_type = ff::AV_PICTURE_TYPE_NONE;
                    (*p).color_range = self.cfg.color.tags.range as _;
                    (*p).colorspace = self.cfg.color.tags.space as _;
                    (*p).color_primaries = self.cfg.color.tags.primaries as _;
                    (*p).color_trc = self.cfg.color.tags.trc as _;
                    p
                }
                None => {
                    self.flushed = true;
                    ptr::null_mut()
                }
            };
            let ret = ff::avcodec_send_frame(self.ctx, fp);
            if ret < 0 && ret != AVERR_EOF {
                return Err(ff_err("avcodec_send_frame", ret).into());
            }
            loop {
                let r = ff::avcodec_receive_packet(self.ctx, self.pkt.as_ptr());
                if r == AVERR_EAGAIN || r == AVERR_EOF {
                    break;
                }
                if r < 0 {
                    return Err(ff_err("avcodec_receive_packet", r).into());
                }
                let res = sink(&mut self.pkt);
                self.pkt.unref();
                res?;
            }
        }
        Ok(())
    }
}

impl Drop for VideoEncoder {
    fn drop(&mut self) {
        unsafe { ff::avcodec_free_context(&mut self.ctx) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_parse() {
        assert_eq!(VideoCodec::parse("HEVC"), Some(VideoCodec::Hevc));
        assert_eq!(VideoCodec::parse("h265"), Some(VideoCodec::Hevc));
        assert_eq!(VideoCodec::parse("h264"), Some(VideoCodec::H264));
        assert_eq!(VideoCodec::parse("av1"), Some(VideoCodec::Av1));
        assert_eq!(VideoCodec::parse("vp9"), None);
    }

    #[test]
    fn av1_needs_ada() {
        assert!(!nvenc_supports(VideoCodec::Av1, 8, 6)); // RTX 30
        assert!(nvenc_supports(VideoCodec::Av1, 8, 9)); // RTX 40
        assert!(nvenc_supports(VideoCodec::Av1, 12, 0)); // RTX 50
        assert!(nvenc_supports(VideoCodec::Hevc, 7, 5));
    }
}
