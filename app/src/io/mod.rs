//! FFmpeg based video I/O: demux + NVDEC decode, NVENC encode + mux.
//!
//! All raw FFmpeg calls are confined to this module; the rest of the app only sees the small
//! RAII types defined here. Frames stay in VRAM (CUDA hw frames) end to end.
#![allow(dead_code)]

pub mod decode;
pub mod encode;
pub mod mux;
pub mod probe;

use rsmpeg::ffi as ff;
use std::ffi::{c_int, CString};
use std::ptr;

pub use ff::AVRational;

pub const NOPTS: i64 = i64::MIN;
pub const AVERR_EOF: i32 = -541478725; // FFERRTAG('E','O','F',' ')
pub const AVERR_EAGAIN: i32 = -11;
pub const AVERR_INVALIDDATA: i32 = -1094995529; // FFERRTAG('I','N','D','A')
pub const AVERR_ENOMEM: i32 = -12;

#[derive(Debug, thiserror::Error)]
pub enum IoError {
    #[error("{what}: {msg} (ffmpeg error {code})")]
    Ff { what: String, code: i32, msg: String },
    /// Codec / pixel format / container the app cannot handle (exit code 6).
    #[error("{0}")]
    Unsupported(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, IoError>;

pub fn av_strerror(code: i32) -> String {
    let mut buf = [0i8; 128];
    unsafe {
        ff::av_strerror(code, buf.as_mut_ptr(), buf.len());
        let bytes: Vec<u8> = buf.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

pub fn ff_err(what: &str, code: i32) -> IoError {
    IoError::Ff { what: what.to_string(), code, msg: av_strerror(code) }
}

/// Turn a negative FFmpeg return value into an error.
#[inline]
pub fn ffcheck(ret: c_int, what: &str) -> Result<c_int> {
    if ret < 0 {
        Err(ff_err(what, ret))
    } else {
        Ok(ret)
    }
}

pub fn cstring(s: &str) -> Result<CString> {
    CString::new(s).map_err(|_| IoError::Other(format!("string contains NUL: {s:?}")))
}

pub fn q2d(q: AVRational) -> f64 {
    if q.den == 0 {
        0.0
    } else {
        q.num as f64 / q.den as f64
    }
}

/// Rescale `v` from timebase `from` to `to` (round to nearest, passing NOPTS through).
pub fn rescale_q(v: i64, from: AVRational, to: AVRational) -> i64 {
    if v == NOPTS {
        return v;
    }
    unsafe { ff::av_rescale_q(v, from, to) }
}

// ---------------------------------------------------------------------------------------
// RAII wrappers
// ---------------------------------------------------------------------------------------

/// Owned `AVFrame`.
pub struct Frame {
    ptr: *mut ff::AVFrame,
}
unsafe impl Send for Frame {}

impl Frame {
    pub fn alloc() -> Result<Frame> {
        let p = unsafe { ff::av_frame_alloc() };
        if p.is_null() {
            return Err(IoError::Other("av_frame_alloc failed".into()));
        }
        Ok(Frame { ptr: p })
    }
    #[inline]
    pub fn as_ptr(&self) -> *mut ff::AVFrame {
        self.ptr
    }
    pub fn unref(&mut self) {
        unsafe { ff::av_frame_unref(self.ptr) }
    }
}
impl Drop for Frame {
    fn drop(&mut self) {
        unsafe { ff::av_frame_free(&mut self.ptr) }
    }
}

/// Owned `AVPacket`.
pub struct Packet {
    ptr: *mut ff::AVPacket,
}
unsafe impl Send for Packet {}

impl Packet {
    pub fn alloc() -> Result<Packet> {
        let p = unsafe { ff::av_packet_alloc() };
        if p.is_null() {
            return Err(IoError::Other("av_packet_alloc failed".into()));
        }
        Ok(Packet { ptr: p })
    }
    #[inline]
    pub fn as_ptr(&self) -> *mut ff::AVPacket {
        self.ptr
    }
    pub fn unref(&mut self) {
        unsafe { ff::av_packet_unref(self.ptr) }
    }
}
impl Drop for Packet {
    fn drop(&mut self) {
        unsafe { ff::av_packet_free(&mut self.ptr) }
    }
}

/// Owned reference to an `AVBufferRef`.
pub struct BufRef {
    ptr: *mut ff::AVBufferRef,
}
unsafe impl Send for BufRef {}
unsafe impl Sync for BufRef {}

impl BufRef {
    /// Take ownership of an existing reference.
    pub unsafe fn from_raw(ptr: *mut ff::AVBufferRef) -> BufRef {
        BufRef { ptr }
    }
    #[inline]
    pub fn as_ptr(&self) -> *mut ff::AVBufferRef {
        self.ptr
    }
    /// A new reference (for handing to an FFmpeg context that takes ownership).
    pub fn new_ref(&self) -> *mut ff::AVBufferRef {
        unsafe { ff::av_buffer_ref(self.ptr) }
    }
}
impl Clone for BufRef {
    fn clone(&self) -> BufRef {
        BufRef { ptr: unsafe { ff::av_buffer_ref(self.ptr) } }
    }
}
impl Drop for BufRef {
    fn drop(&mut self) {
        unsafe { ff::av_buffer_unref(&mut self.ptr) }
    }
}

/// CUDA hardware device context created on the device's primary context, so FFmpeg
/// (NVDEC/NVENC) and fb_native share one CUDA context.
#[derive(Clone)]
pub struct CudaDevice {
    dev: BufRef,
    pub index: i32,
}

/// `AV_CUDA_USE_PRIMARY_CONTEXT` (hwcontext_cuda.h).
const AV_CUDA_USE_PRIMARY_CONTEXT: c_int = 1;

static SHARED_DEVICE: std::sync::Mutex<Option<CudaDevice>> = std::sync::Mutex::new(None);

impl CudaDevice {
    /// Process-wide device context. It MUST be created before `native::init` first touches the
    /// device: FFmpeg sets the primary context's flags (blocking sync) and refuses to attach to a
    /// primary context that the CUDA runtime already activated with other flags.
    pub fn shared(index: i32) -> Result<CudaDevice> {
        let mut g = SHARED_DEVICE.lock().unwrap();
        if let Some(d) = g.as_ref() {
            if d.index == index {
                return Ok(d.clone());
            }
        }
        let d = CudaDevice::new(index)?;
        *g = Some(d.clone());
        Ok(d)
    }

    pub fn new(index: i32) -> Result<CudaDevice> {
        let name = cstring(&index.to_string())?;
        let mut p: *mut ff::AVBufferRef = ptr::null_mut();
        let ret = unsafe {
            ff::av_hwdevice_ctx_create(
                &mut p,
                ff::AV_HWDEVICE_TYPE_CUDA,
                name.as_ptr(),
                ptr::null_mut(),
                AV_CUDA_USE_PRIMARY_CONTEXT,
            )
        };
        ffcheck(ret, "av_hwdevice_ctx_create(cuda)")?;
        Ok(CudaDevice { dev: unsafe { BufRef::from_raw(p) }, index })
    }
    pub fn as_ptr(&self) -> *mut ff::AVBufferRef {
        self.dev.as_ptr()
    }
}

/// Silence FFmpeg's console logging below `level` (AV_LOG_*): 16 = error, 24 = warning.
pub fn set_log_level(level: i32) {
    unsafe { ff::av_log_set_level(level) }
}

/// Snap a measured frame rate to the nearest well-known rate when within 0.05%.
pub fn snap_frame_rate(r: AVRational) -> AVRational {
    if r.num <= 0 || r.den <= 0 {
        return r;
    }
    const KNOWN: [(i32, i32); 14] = [
        (24000, 1001),
        (24, 1),
        (25, 1),
        (30000, 1001),
        (30, 1),
        (48, 1),
        (50, 1),
        (60000, 1001),
        (60, 1),
        (90, 1),
        (100, 1),
        (120000, 1001),
        (120, 1),
        (240, 1),
    ];
    let v = r.num as f64 / r.den as f64;
    for (n, d) in KNOWN {
        let k = n as f64 / d as f64;
        if ((v - k) / k).abs() < 0.0005 {
            return AVRational { num: n, den: d };
        }
    }
    r
}

/// Best effort for the container fps when `avg_frame_rate` is missing.
pub fn pick_frame_rate(avg: AVRational, r: AVRational) -> Option<AVRational> {
    let ok = |q: AVRational| q.num > 0 && q.den > 0;
    if ok(avg) {
        Some(snap_frame_rate(avg))
    } else if ok(r) {
        Some(snap_frame_rate(r))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snap_rates() {
        let s = snap_frame_rate(AVRational { num: 2997, den: 100 });
        assert_eq!((s.num, s.den), (30000, 1001));
        let s = snap_frame_rate(AVRational { num: 30, den: 1 });
        assert_eq!((s.num, s.den), (30, 1));
        let s = snap_frame_rate(AVRational { num: 12345, den: 1000 });
        assert_eq!((s.num, s.den), (12345, 1000));
    }

    #[test]
    fn pick_rate() {
        let z = AVRational { num: 0, den: 0 };
        let r = AVRational { num: 25, den: 1 };
        assert_eq!(pick_frame_rate(z, r).map(|q| q.num), Some(25));
        assert!(pick_frame_rate(z, z).is_none());
    }
}
