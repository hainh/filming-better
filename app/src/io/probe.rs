//! Demux-level information about an input file.

use super::*;
use crate::native::{Matrix, Range};
use std::path::Path;

/// Owned input `AVFormatContext`.
pub struct InputCtx {
    ptr: *mut ff::AVFormatContext,
}
unsafe impl Send for InputCtx {}

impl InputCtx {
    pub fn open(path: &Path) -> Result<InputCtx> {
        let cpath = cstring(&path.to_string_lossy())?;
        let mut ctx: *mut ff::AVFormatContext = ptr::null_mut();
        // FFmpeg would print its own error line; ours (with the reason) is enough.
        let old_level = unsafe { ff::av_log_get_level() };
        unsafe { ff::av_log_set_level(ff::AV_LOG_FATAL as i32) };
        let ret = unsafe { ff::avformat_open_input(&mut ctx, cpath.as_ptr(), ptr::null(), ptr::null_mut()) };
        unsafe { ff::av_log_set_level(old_level) };
        if ret < 0 {
            return Err(ff_err(&format!("cannot open '{}'", path.display()), ret));
        }
        let ic = InputCtx { ptr: ctx };
        let ret = unsafe { ff::avformat_find_stream_info(ic.ptr, ptr::null_mut()) };
        ffcheck(ret, "avformat_find_stream_info")?;
        Ok(ic)
    }
    #[inline]
    pub fn as_ptr(&self) -> *mut ff::AVFormatContext {
        self.ptr
    }
    pub fn nb_streams(&self) -> usize {
        unsafe { (*self.ptr).nb_streams as usize }
    }
    pub fn stream(&self, i: usize) -> *mut ff::AVStream {
        unsafe { *(*self.ptr).streams.add(i) }
    }
}
impl Drop for InputCtx {
    fn drop(&mut self) {
        unsafe { ff::avformat_close_input(&mut self.ptr) }
    }
}

/// Colour description of a stream (raw FFmpeg enum values).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorTags {
    pub space: i32,
    pub range: i32,
    pub primaries: i32,
    pub trc: i32,
}

pub const AVCOL_SPC_RGB: i32 = 0;
pub const AVCOL_SPC_BT709: i32 = 1;
pub const AVCOL_SPC_UNSPECIFIED: i32 = 2;
pub const AVCOL_SPC_BT470BG: i32 = 5;
pub const AVCOL_SPC_SMPTE170M: i32 = 6;
pub const AVCOL_SPC_SMPTE240M: i32 = 7;
pub const AVCOL_SPC_BT2020_NCL: i32 = 9;
pub const AVCOL_RANGE_MPEG: i32 = 1;
pub const AVCOL_RANGE_JPEG: i32 = 2;
pub const AVCOL_PRI_BT709: i32 = 1;
pub const AVCOL_PRI_SMPTE170M: i32 = 6;
pub const AVCOL_PRI_BT2020: i32 = 9;
pub const AVCOL_TRC_BT709: i32 = 1;
pub const AVCOL_TRC_SMPTE170M: i32 = 6;
pub const AVCOL_TRC_SMPTE2084: i32 = 16;
pub const AVCOL_TRC_ARIB_STD_B67: i32 = 18;

impl ColorTags {
    pub fn is_hdr(&self) -> bool {
        self.trc == AVCOL_TRC_SMPTE2084 || self.trc == AVCOL_TRC_ARIB_STD_B67
    }
}

/// Colour conversion parameters resolved from tags + heuristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorSpec {
    pub matrix: Matrix,
    pub range: Range,
    /// Tags to write on the output (always concrete).
    pub tags: ColorTags,
}

/// Resolve colour handling. Unspecified matrix: BT.709 for HD (>= 720 lines or >= 1280 wide),
/// BT.601 otherwise. Unspecified range: limited.
pub fn resolve_color(tags: ColorTags, width: i32, height: i32) -> ColorSpec {
    let matrix = match tags.space {
        AVCOL_SPC_BT709 => Matrix::Bt709,
        AVCOL_SPC_BT470BG | AVCOL_SPC_SMPTE170M | AVCOL_SPC_SMPTE240M => Matrix::Bt601,
        AVCOL_SPC_BT2020_NCL => Matrix::Bt2020Nc,
        _ => {
            if height >= 720 || width >= 1280 {
                Matrix::Bt709
            } else {
                Matrix::Bt601
            }
        }
    };
    let range = if tags.range == AVCOL_RANGE_JPEG { Range::Full } else { Range::Limited };
    let (space, pri, trc) = match matrix {
        Matrix::Bt709 => (AVCOL_SPC_BT709, AVCOL_PRI_BT709, AVCOL_TRC_BT709),
        Matrix::Bt601 => (
            if tags.space == AVCOL_SPC_BT470BG { AVCOL_SPC_BT470BG } else { AVCOL_SPC_SMPTE170M },
            if tags.primaries == 5 { 5 } else { AVCOL_PRI_SMPTE170M },
            AVCOL_TRC_SMPTE170M,
        ),
        Matrix::Bt2020Nc => (AVCOL_SPC_BT2020_NCL, AVCOL_PRI_BT2020, AVCOL_TRC_BT709),
    };
    let unspec = |v: i32| v == 0 || v == 2;
    ColorSpec {
        matrix,
        range,
        tags: ColorTags {
            space,
            range: if range == Range::Full { AVCOL_RANGE_JPEG } else { AVCOL_RANGE_MPEG },
            primaries: if unspec(tags.primaries) { pri } else { tags.primaries },
            trc: if unspec(tags.trc) { trc } else { tags.trc },
        },
    }
}

#[derive(Debug, Clone)]
pub struct VideoInfo {
    pub stream_index: usize,
    pub codec_id: i32,
    pub codec_name: String,
    pub profile: i32,
    pub width: i32,
    pub height: i32,
    pub pix_fmt: i32,
    pub pix_fmt_name: String,
    pub bit_depth: u32,
    pub time_base: AVRational,
    /// Stream start time in `time_base` units (0 when unknown).
    pub start_time: i64,
    pub avg_frame_rate: AVRational,
    pub r_frame_rate: AVRational,
    /// Resolved CFR grid rate (snapped), None if the container gives none.
    pub fps: Option<AVRational>,
    pub nb_frames: i64,
    pub duration_s: f64,
    pub color: ColorTags,
    pub sample_aspect_ratio: AVRational,
    /// Clockwise rotation in degrees from the display matrix (0 when absent).
    pub rotation: f64,
    pub display_matrix: Option<[u8; 36]>,
    pub field_order: i32,
}

impl VideoInfo {
    /// Estimated number of frames (container count, else duration * fps).
    pub fn estimated_frames(&self) -> u64 {
        if self.nb_frames > 0 {
            return self.nb_frames as u64;
        }
        match self.fps {
            Some(f) if self.duration_s > 0.0 => (self.duration_s * q2d(f)).round() as u64,
            _ => 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MediaInfo {
    pub path: std::path::PathBuf,
    pub format_name: String,
    pub duration_s: f64,
    pub bit_rate: i64,
    pub video: VideoInfo,
    pub audio_streams: usize,
    pub subtitle_streams: usize,
    pub other_streams: usize,
}

fn cstr_lossy(p: *const std::ffi::c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        unsafe { std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned() }
    }
}

pub fn pix_fmt_name(fmt: i32) -> String {
    cstr_lossy(unsafe { ff::av_get_pix_fmt_name(fmt) })
}

pub fn video_info_from_stream(ic: &InputCtx, idx: usize) -> VideoInfo {
    unsafe {
        let st = ic.stream(idx);
        let par = (*st).codecpar;
        let desc = ff::av_pix_fmt_desc_get((*par).format);
        let bit_depth = if !desc.is_null() {
            (*desc).comp[0].depth as u32
        } else if (*par).bits_per_raw_sample > 0 {
            (*par).bits_per_raw_sample as u32
        } else {
            8
        };
        let tb = (*st).time_base;
        let start = if (*st).start_time == NOPTS { 0 } else { (*st).start_time };
        let dur_s = if (*st).duration != NOPTS && (*st).duration > 0 {
            (*st).duration as f64 * q2d(tb)
        } else if (*ic.as_ptr()).duration != NOPTS && (*ic.as_ptr()).duration > 0 {
            (*ic.as_ptr()).duration as f64 / ff::AV_TIME_BASE as f64
        } else {
            0.0
        };
        let sd = ff::av_packet_side_data_get(
            (*par).coded_side_data,
            (*par).nb_coded_side_data,
            ff::AV_PKT_DATA_DISPLAYMATRIX,
        );
        let (rotation, display_matrix) = if !sd.is_null() && (*sd).size >= 36 {
            let mut m = [0u8; 36];
            ptr::copy_nonoverlapping((*sd).data, m.as_mut_ptr(), 36);
            let rot = -ff::av_display_rotation_get((*sd).data as *const i32);
            let rot = if rot.is_nan() { 0.0 } else { rot };
            (rot, Some(m))
        } else {
            (0.0, None)
        };
        let codec_name = cstr_lossy(ff::avcodec_get_name((*par).codec_id));
        let pix_name = if desc.is_null() { String::new() } else { cstr_lossy((*desc).name) };
        VideoInfo {
            stream_index: idx,
            codec_id: (*par).codec_id as i32,
            codec_name,
            profile: (*par).profile,
            width: (*par).width,
            height: (*par).height,
            pix_fmt: (*par).format,
            pix_fmt_name: pix_name,
            bit_depth,
            time_base: tb,
            start_time: start,
            avg_frame_rate: (*st).avg_frame_rate,
            r_frame_rate: (*st).r_frame_rate,
            fps: pick_frame_rate((*st).avg_frame_rate, (*st).r_frame_rate),
            nb_frames: (*st).nb_frames,
            duration_s: dur_s,
            color: ColorTags {
                space: (*par).color_space as i32,
                range: (*par).color_range as i32,
                primaries: (*par).color_primaries as i32,
                trc: (*par).color_trc as i32,
            },
            sample_aspect_ratio: (*st).sample_aspect_ratio,
            rotation,
            display_matrix,
            field_order: (*par).field_order as i32,
        }
    }
}

pub fn find_video_stream(ic: &InputCtx) -> Result<usize> {
    let idx = unsafe {
        ff::av_find_best_stream(ic.as_ptr(), ff::AVMEDIA_TYPE_VIDEO, -1, -1, ptr::null_mut(), 0)
    };
    if idx < 0 {
        return Err(IoError::Unsupported("the file has no video stream".into()));
    }
    Ok(idx as usize)
}

pub fn media_info(ic: &InputCtx, path: &Path) -> Result<MediaInfo> {
    let vidx = find_video_stream(ic)?;
    let video = video_info_from_stream(ic, vidx);
    let (mut a, mut s, mut o) = (0, 0, 0);
    for i in 0..ic.nb_streams() {
        if i == vidx {
            continue;
        }
        let t = unsafe { (*(*ic.stream(i)).codecpar).codec_type };
        match t {
            ff::AVMEDIA_TYPE_AUDIO => a += 1,
            ff::AVMEDIA_TYPE_SUBTITLE => s += 1,
            _ => o += 1,
        }
    }
    let fc = ic.as_ptr();
    let (fmt_name, dur, br) = unsafe {
        let n = if (*fc).iformat.is_null() { String::new() } else { cstr_lossy((*(*fc).iformat).name) };
        let d = if (*fc).duration > 0 { (*fc).duration as f64 / ff::AV_TIME_BASE as f64 } else { video.duration_s };
        (n, d, (*fc).bit_rate)
    };
    Ok(MediaInfo {
        path: path.to_path_buf(),
        format_name: fmt_name,
        duration_s: dur,
        bit_rate: br,
        video,
        audio_streams: a,
        subtitle_streams: s,
        other_streams: o,
    })
}

pub fn probe(path: &Path) -> Result<MediaInfo> {
    let ic = InputCtx::open(path)?;
    media_info(&ic, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_resolution() {
        let t = ColorTags { space: 2, range: 0, primaries: 2, trc: 2 };
        let c = resolve_color(t, 576, 1024);
        assert_eq!(c.matrix, Matrix::Bt709);
        assert_eq!(c.range, Range::Limited);
        assert_eq!(c.tags.space, AVCOL_SPC_BT709);
        let c = resolve_color(t, 640, 480);
        assert_eq!(c.matrix, Matrix::Bt601);
        assert_eq!(c.tags.trc, AVCOL_TRC_SMPTE170M);
        let t = ColorTags { space: AVCOL_SPC_BT709, range: AVCOL_RANGE_JPEG, primaries: 1, trc: 1 };
        let c = resolve_color(t, 640, 480);
        assert_eq!(c.matrix, Matrix::Bt709);
        assert_eq!(c.range, Range::Full);
    }
}
