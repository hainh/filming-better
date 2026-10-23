//! Demux + decode. Frames come out as CUDA hardware frames (NV12 / P010) that stay in VRAM:
//! NVDEC through FFmpeg's hwaccel when possible, otherwise CPU decode + `hwframe_transfer`.

use super::probe::*;
use super::*;
use crate::native::{YuvFormat, YuvFrame};
use std::collections::VecDeque;
use std::path::Path;

/// A non-video packet (audio / subtitle / ...) read while looking for video packets.
pub struct SidePacket {
    pub stream_index: usize,
    pub pkt: Packet,
}

/// A decoded picture in VRAM. Holds the underlying AVFrame (and thereby the decoder's
/// surface) until dropped, so drop it as soon as the conversion kernel finished.
pub struct GpuFrame {
    frame: Frame,
    pub format: YuvFormat,
    pub width: i32,
    pub height: i32,
    /// Presentation timestamp in the stream time base (`NOPTS` when unknown).
    pub pts: i64,
    pub color: ColorTags,
}

impl GpuFrame {
    pub fn yuv(&self) -> YuvFrame {
        unsafe {
            let f = self.frame.as_ptr();
            YuvFrame {
                luma: (*f).data[0] as u64,
                luma_pitch: (*f).linesize[0] as usize,
                chroma: (*f).data[1] as u64,
                chroma_pitch: (*f).linesize[1] as usize,
                width: self.width,
                height: self.height,
                format: self.format,
            }
        }
    }
}

unsafe extern "C" fn get_hw_format(_ctx: *mut ff::AVCodecContext, fmts: *const ff::AVPixelFormat) -> ff::AVPixelFormat {
    let mut p = fmts;
    let mut first_sw = ff::AV_PIX_FMT_NONE;
    while *p != ff::AV_PIX_FMT_NONE {
        if *p == ff::AV_PIX_FMT_CUDA {
            return ff::AV_PIX_FMT_CUDA;
        }
        if first_sw == ff::AV_PIX_FMT_NONE {
            let d = ff::av_pix_fmt_desc_get(*p);
            if !d.is_null() && ((*d).flags & ff::AV_PIX_FMT_FLAG_HWACCEL as u64) == 0 {
                first_sw = *p;
            }
        }
        p = p.add(1);
    }
    first_sw
}

fn codec_supports_cuda(codec: *const ff::AVCodec) -> bool {
    unsafe {
        let mut i = 0;
        loop {
            let cfg = ff::avcodec_get_hw_config(codec, i);
            if cfg.is_null() {
                return false;
            }
            if ((*cfg).methods & ff::AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX as i32) != 0
                && (*cfg).device_type == ff::AV_HWDEVICE_TYPE_CUDA
            {
                return true;
            }
            i += 1;
        }
    }
}

/// Uploads software frames to CUDA frames (NV12 / P010) when NVDEC cannot decode the stream.
struct Uploader {
    device: CudaDevice,
    sws: *mut ff::SwsContext,
    frames_ref: Option<BufRef>,
    key: (i32, i32, i32),
    tmp: Option<Frame>,
}

impl Uploader {
    fn new(device: CudaDevice) -> Uploader {
        Uploader { device, sws: ptr::null_mut(), frames_ref: None, key: (0, 0, -1), tmp: None }
    }

    fn ensure_pool(&mut self, w: i32, h: i32, fmt: i32) -> Result<()> {
        if self.frames_ref.is_some() && self.key == (w, h, fmt) {
            return Ok(());
        }
        unsafe {
            let r = ff::av_hwframe_ctx_alloc(self.device.as_ptr());
            if r.is_null() {
                return Err(IoError::Other("av_hwframe_ctx_alloc failed".into()));
            }
            let fr = BufRef::from_raw(r);
            let fc = (*r).data as *mut ff::AVHWFramesContext;
            (*fc).format = ff::AV_PIX_FMT_CUDA;
            (*fc).sw_format = fmt;
            (*fc).width = w;
            (*fc).height = h;
            (*fc).initial_pool_size = 0;
            ffcheck(ff::av_hwframe_ctx_init(r), "av_hwframe_ctx_init(upload)")?;
            self.frames_ref = Some(fr);
            let mut t = Frame::alloc()?;
            (*t.as_ptr()).format = fmt;
            (*t.as_ptr()).width = w;
            (*t.as_ptr()).height = h;
            ffcheck(ff::av_frame_get_buffer(t.as_ptr(), 0), "av_frame_get_buffer")?;
            self.tmp = Some(t);
            self.key = (w, h, fmt);
        }
        Ok(())
    }

    fn upload(&mut self, src: &Frame) -> Result<(Frame, YuvFormat)> {
        unsafe {
            let s = src.as_ptr();
            let (w, h) = ((*s).width, (*s).height);
            let desc = ff::av_pix_fmt_desc_get((*s).format);
            if desc.is_null() {
                return Err(IoError::Unsupported("decoder produced an unknown pixel format".into()));
            }
            let depth = (*desc).comp[0].depth;
            let (dst_fmt, yf) = if depth > 8 {
                (ff::AV_PIX_FMT_P010LE, YuvFormat::P010)
            } else {
                (ff::AV_PIX_FMT_NV12, YuvFormat::Nv12)
            };
            self.ensure_pool(w, h, dst_fmt)?;
            let tmp = self.tmp.as_ref().unwrap().as_ptr();
            ffcheck(ff::av_frame_make_writable(tmp), "av_frame_make_writable")?;

            self.sws = ff::sws_getCachedContext(
                self.sws,
                w,
                h,
                (*s).format,
                w,
                h,
                dst_fmt,
                (ff::SWS_BILINEAR | ff::SWS_ACCURATE_RND) as i32,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null(),
            );
            if self.sws.is_null() {
                return Err(IoError::Unsupported(format!(
                    "cannot convert pixel format {} to NV12/P010",
                    super::probe::pix_fmt_name((*s).format)
                )));
            }
            // Same range on both sides: only repack / resample chroma, never rescale levels.
            let full = (*s).color_range == ff::AVCOL_RANGE_JPEG || is_yuvj((*s).format);
            let coeffs = ff::sws_getCoefficients(ff::SWS_CS_DEFAULT as i32) as *const c_int;
            ff::sws_setColorspaceDetails(self.sws, coeffs, full as c_int, coeffs, full as c_int, 0, 1 << 16, 1 << 16);
            ff::sws_scale(
                self.sws,
                (*s).data.as_ptr() as *const *const u8,
                (*s).linesize.as_ptr(),
                0,
                h,
                (*tmp).data.as_mut_ptr(),
                (*tmp).linesize.as_mut_ptr(),
            );
            let mut out = Frame::alloc()?;
            ffcheck(
                ff::av_hwframe_get_buffer(self.frames_ref.as_ref().unwrap().as_ptr(), out.as_ptr(), 0),
                "av_hwframe_get_buffer(upload)",
            )?;
            ffcheck(ff::av_hwframe_transfer_data(out.as_ptr(), tmp, 0), "av_hwframe_transfer_data")?;
            ff::av_frame_copy_props(out.as_ptr(), s);
            (*out.as_ptr()).color_range = if full { ff::AVCOL_RANGE_JPEG } else { ff::AVCOL_RANGE_MPEG };
            let _ = &mut out;
            Ok((out, yf))
        }
    }
}

impl Drop for Uploader {
    fn drop(&mut self) {
        unsafe { ff::sws_freeContext(self.sws) }
    }
}

fn is_yuvj(fmt: i32) -> bool {
    matches!(
        fmt,
        x if x == ff::AV_PIX_FMT_YUVJ420P || x == ff::AV_PIX_FMT_YUVJ422P || x == ff::AV_PIX_FMT_YUVJ444P
    )
}

/// Video decoder + demuxer for one input file.
pub struct VideoSource {
    ic: InputCtx,
    pub info: VideoInfo,
    dec: *mut ff::AVCodecContext,
    pub hw: bool,
    pkt: Packet,
    frm: Frame,
    eof_sent: bool,
    eof: bool,
    forward: Vec<bool>,
    side: VecDeque<SidePacket>,
    pending: VecDeque<GpuFrame>,
    upload: Uploader,
}

unsafe impl Send for VideoSource {}

impl VideoSource {
    /// Open `path`. `prefer_hw`: try NVDEC first. `forward_side`: queue audio/subtitle packets
    /// in [`take_side`](Self::take_side) (for remuxing).
    pub fn open(path: &Path, device: &CudaDevice, prefer_hw: bool, forward_side: bool) -> Result<VideoSource> {
        let ic = InputCtx::open(path)?;
        let vidx = find_video_stream(&ic)?;
        let info = video_info_from_stream(&ic, vidx);
        unsafe {
            let st = ic.stream(vidx);
            let par = (*st).codecpar;
            let codec = ff::avcodec_find_decoder((*par).codec_id);
            if codec.is_null() {
                return Err(IoError::Unsupported(format!(
                    "no decoder for video codec '{}' in this FFmpeg build",
                    info.codec_name
                )));
            }
            let hw = prefer_hw && codec_supports_cuda(codec);
            let ctx = ff::avcodec_alloc_context3(codec);
            if ctx.is_null() {
                return Err(IoError::Other("avcodec_alloc_context3 failed".into()));
            }
            let guard = |ctx: *mut ff::AVCodecContext| {
                let mut c = ctx;
                ff::avcodec_free_context(&mut c);
            };
            if let Err(e) = ffcheck(ff::avcodec_parameters_to_context(ctx, par), "avcodec_parameters_to_context") {
                guard(ctx);
                return Err(e);
            }
            (*ctx).pkt_timebase = (*st).time_base;
            if hw {
                (*ctx).hw_device_ctx = device.dev_ref();
                (*ctx).get_format = Some(get_hw_format);
                (*ctx).extra_hw_frames = 3;
            }
            let ret = ff::avcodec_open2(ctx, codec, ptr::null_mut());
            if ret < 0 {
                guard(ctx);
                return Err(ff_err(&format!("cannot open decoder '{}'", info.codec_name), ret));
            }
            let mut forward = vec![false; ic.nb_streams()];
            if forward_side {
                for (i, f) in forward.iter_mut().enumerate() {
                    let t = (*(*ic.stream(i)).codecpar).codec_type;
                    *f = i != vidx && (t == ff::AVMEDIA_TYPE_AUDIO || t == ff::AVMEDIA_TYPE_SUBTITLE);
                }
            }
            Ok(VideoSource {
                ic,
                info,
                dec: ctx,
                hw,
                pkt: Packet::alloc()?,
                frm: Frame::alloc()?,
                eof_sent: false,
                eof: false,
                forward,
                side: VecDeque::new(),
                pending: VecDeque::new(),
                upload: Uploader::new(device.clone()),
            })
        }
    }

    pub fn input(&self) -> &InputCtx {
        &self.ic
    }

    /// Drain queued non-video packets.
    pub fn take_side(&mut self) -> std::collections::vec_deque::Drain<'_, SidePacket> {
        self.side.drain(..)
    }

    /// Put a frame back so the next [`next_frame`](Self::next_frame) returns it.
    pub fn push_back(&mut self, f: GpuFrame) {
        self.pending.push_front(f);
    }

    /// Seek (backwards to the previous keyframe) to `secs` from the start of the stream.
    pub fn seek(&mut self, secs: f64) -> Result<()> {
        let tb = q2d(self.info.time_base);
        let ts = self.info.start_time + (secs / tb).round() as i64;
        unsafe {
            ffcheck(
                ff::av_seek_frame(self.ic.as_ptr(), self.info.stream_index as i32, ts, ff::AVSEEK_FLAG_BACKWARD as i32),
                "av_seek_frame",
            )?;
            ff::avcodec_flush_buffers(self.dec);
        }
        self.eof = false;
        self.eof_sent = false;
        self.pending.clear();
        self.side.clear();
        Ok(())
    }

    pub fn next_frame(&mut self) -> Result<Option<GpuFrame>> {
        if let Some(f) = self.pending.pop_front() {
            return Ok(Some(f));
        }
        if self.eof {
            return Ok(None);
        }
        loop {
            let ret = unsafe { ff::avcodec_receive_frame(self.dec, self.frm.as_ptr()) };
            if ret >= 0 {
                return self.wrap_frame().map(Some);
            }
            if ret == AVERR_EOF {
                self.eof = true;
                return Ok(None);
            }
            if ret != AVERR_EAGAIN {
                return Err(ff_err("avcodec_receive_frame", ret));
            }
            if self.eof_sent {
                self.eof = true;
                return Ok(None);
            }
            let r = unsafe { ff::av_read_frame(self.ic.as_ptr(), self.pkt.as_ptr()) };
            if r < 0 {
                if r != AVERR_EOF {
                    tracing::warn!("av_read_frame: {} - treating as end of input", av_strerror(r));
                }
                unsafe { ff::avcodec_send_packet(self.dec, ptr::null()) };
                self.eof_sent = true;
                continue;
            }
            let si = unsafe { (*self.pkt.as_ptr()).stream_index } as usize;
            if si == self.info.stream_index {
                let s = unsafe { ff::avcodec_send_packet(self.dec, self.pkt.as_ptr()) };
                self.pkt.unref();
                if s < 0 && s != AVERR_EAGAIN && s != AVERR_EOF {
                    if s == AVERR_INVALIDDATA {
                        tracing::warn!("skipping undecodable video packet");
                        continue;
                    }
                    return Err(ff_err("avcodec_send_packet", s));
                }
            } else if si < self.forward.len() && self.forward[si] {
                let np = Packet::alloc()?;
                unsafe { ff::av_packet_move_ref(np.as_ptr(), self.pkt.as_ptr()) };
                self.side.push_back(SidePacket { stream_index: si, pkt: np });
            } else {
                self.pkt.unref();
            }
        }
    }

    fn wrap_frame(&mut self) -> Result<GpuFrame> {
        unsafe {
            let f = self.frm.as_ptr();
            let pts = if (*f).best_effort_timestamp != NOPTS { (*f).best_effort_timestamp } else { (*f).pts };
            let (frame, format) = if (*f).format == ff::AV_PIX_FMT_CUDA {
                let hwf = (*f).hw_frames_ctx;
                let sw = if hwf.is_null() {
                    -1
                } else {
                    (*((*hwf).data as *mut ff::AVHWFramesContext)).sw_format
                };
                let format = if sw == ff::AV_PIX_FMT_NV12 {
                    YuvFormat::Nv12
                } else if sw == ff::AV_PIX_FMT_P010LE {
                    YuvFormat::P010
                } else {
                    self.frm.unref();
                    return Err(IoError::Unsupported(format!(
                        "NVDEC produced pixel format '{}' (only 4:2:0 8/10-bit is supported); retry with CPU decoding",
                        super::probe::pix_fmt_name(sw)
                    )));
                };
                let out = Frame::alloc()?;
                ff::av_frame_move_ref(out.as_ptr(), f);
                (out, format)
            } else {
                let r = self.upload.upload(&self.frm);
                self.frm.unref();
                r?
            };
            let o = frame.as_ptr();
            let color = ColorTags {
                space: (*o).colorspace as i32,
                range: (*o).color_range as i32,
                primaries: (*o).color_primaries as i32,
                trc: (*o).color_trc as i32,
            };
            // Some hardware decoders (VP9/AV1 via NVDEC) hand out surfaces rounded up to even sizes;
            // the visible picture is what the stream header says.
            let (mut w, mut h) = ((*o).width, (*o).height);
            let (iw, ih) = (self.info.width, self.info.height);
            if iw > 0 && ih > 0 && w >= iw && h >= ih && w - iw <= 1 && h - ih <= 1 {
                w = iw;
                h = ih;
            }
            Ok(GpuFrame { width: w, height: h, frame, format, pts, color })
        }
    }
}

impl Drop for VideoSource {
    fn drop(&mut self) {
        self.pending.clear();
        unsafe { ff::avcodec_free_context(&mut self.dec) }
    }
}

impl CudaDevice {
    /// New owned reference to the device context (for FFmpeg contexts that take ownership).
    pub fn dev_ref(&self) -> *mut ff::AVBufferRef {
        unsafe { ff::av_buffer_ref(self.as_ptr()) }
    }
}
