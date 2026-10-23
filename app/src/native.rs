//! Safe wrappers around the `fb_native` C API (see `native/include/fb_native.h`).
//!
//! Every handle is wrapped in an RAII type; every status code becomes a [`NativeError`]
//! carrying the numeric code and the message from `fb_last_error()`.
#![allow(dead_code)]

use std::ffi::{c_char, CStr, CString};
use std::path::Path;

#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code, clippy::all)]
mod ffi {
    include!(concat!(env!("OUT_DIR"), "/fb_native_bindings.rs"));
}

// ---------------------------------------------------------------------------------------
// errors
// ---------------------------------------------------------------------------------------

/// Numeric status codes of fb_native (mirrors the `FB_ERR_*` enum).
pub mod code {
    pub const OK: i32 = super::ffi::FB_OK as i32;
    pub const INVALID_ARG: i32 = super::ffi::FB_ERR_INVALID_ARG as i32;
    pub const NO_GPU: i32 = super::ffi::FB_ERR_NO_GPU as i32;
    pub const DRIVER_TOO_OLD: i32 = super::ffi::FB_ERR_DRIVER_TOO_OLD as i32;
    pub const OUT_OF_MEMORY: i32 = super::ffi::FB_ERR_OUT_OF_MEMORY as i32;
    pub const CUDA: i32 = super::ffi::FB_ERR_CUDA as i32;
    pub const TRT_LOAD: i32 = super::ffi::FB_ERR_TRT_LOAD as i32;
    pub const ENGINE: i32 = super::ffi::FB_ERR_ENGINE as i32;
    pub const NOT_INITIALIZED: i32 = super::ffi::FB_ERR_NOT_INITIALIZED as i32;
    pub const UNSUPPORTED: i32 = super::ffi::FB_ERR_UNSUPPORTED as i32;
    pub const IO: i32 = super::ffi::FB_ERR_IO as i32;
    pub const INTERNAL: i32 = super::ffi::FB_ERR_INTERNAL as i32;
}

/// Minimum CUDA driver version required by the native library (1000*major + 10*minor).
pub const MIN_DRIVER_VERSION: i32 = ffi::FB_MIN_DRIVER_VERSION as i32;

#[derive(Debug, Clone, thiserror::Error)]
#[error("{op}: {message} (fb_native code {code})")]
pub struct NativeError {
    pub code: i32,
    pub op: &'static str,
    pub message: String,
    /// Driver version reported by `fb_init` (0 if unknown); set for `DRIVER_TOO_OLD`.
    pub driver_version: i32,
}

impl NativeError {
    pub fn new(code: i32, op: &'static str, message: impl Into<String>) -> Self {
        NativeError { code, op, message: message.into(), driver_version: 0 }
    }
    pub fn is_oom(&self) -> bool {
        self.code == code::OUT_OF_MEMORY
    }
    pub fn is_no_gpu(&self) -> bool {
        self.code == code::NO_GPU
    }
    pub fn is_driver_too_old(&self) -> bool {
        self.code == code::DRIVER_TOO_OLD
    }
    pub fn is_trt_load(&self) -> bool {
        self.code == code::TRT_LOAD
    }
}

pub type Result<T> = std::result::Result<T, NativeError>;

fn last_error() -> String {
    unsafe {
        let p = ffi::fb_last_error();
        if p.is_null() {
            String::new()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

#[inline]
fn check(rc: i32, op: &'static str) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(NativeError::new(rc, op, last_error()))
    }
}

fn cstr_to_string(buf: &[c_char]) -> String {
    let bytes: Vec<u8> = buf.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

// ---------------------------------------------------------------------------------------
// enums
// ---------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Fp32,
    Fp16,
}

impl DType {
    pub fn raw(self) -> i32 {
        match self {
            DType::Fp32 => ffi::FB_DTYPE_FP32 as i32,
            DType::Fp16 => ffi::FB_DTYPE_FP16 as i32,
        }
    }
    pub fn from_raw(v: i32) -> Option<DType> {
        if v == ffi::FB_DTYPE_FP32 as i32 {
            Some(DType::Fp32)
        } else if v == ffi::FB_DTYPE_FP16 as i32 {
            Some(DType::Fp16)
        } else {
            None
        }
    }
    pub fn bytes(self) -> usize {
        match self {
            DType::Fp32 => 4,
            DType::Fp16 => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YuvFormat {
    Nv12,
    P010,
}

impl YuvFormat {
    fn raw(self) -> i32 {
        match self {
            YuvFormat::Nv12 => ffi::FB_FMT_NV12 as i32,
            YuvFormat::P010 => ffi::FB_FMT_P010 as i32,
        }
    }
    pub fn bytes_per_sample(self) -> usize {
        match self {
            YuvFormat::Nv12 => 1,
            YuvFormat::P010 => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Matrix {
    Bt601,
    Bt709,
    Bt2020Nc,
}

impl Matrix {
    fn raw(self) -> i32 {
        match self {
            Matrix::Bt601 => ffi::FB_MATRIX_BT601 as i32,
            Matrix::Bt709 => ffi::FB_MATRIX_BT709 as i32,
            Matrix::Bt2020Nc => ffi::FB_MATRIX_BT2020NC as i32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Range {
    Limited,
    Full,
}

impl Range {
    fn raw(self) -> i32 {
        match self {
            Range::Limited => ffi::FB_RANGE_LIMITED as i32,
            Range::Full => ffi::FB_RANGE_FULL as i32,
        }
    }
}

// ---------------------------------------------------------------------------------------
// device
// ---------------------------------------------------------------------------------------

pub fn api_version() -> i32 {
    unsafe { ffi::fb_api_version() as i32 }
}

/// Initialise the device and activate its primary context for the calling thread.
/// Safe to call repeatedly and from every worker thread. Returns the driver version.
pub fn init(device_index: i32) -> Result<i32> {
    let mut drv: i32 = 0;
    let rc = unsafe { ffi::fb_init(device_index, &mut drv) };
    if rc != 0 {
        let mut e = NativeError::new(rc, "fb_init", last_error());
        e.driver_version = drv;
        return Err(e);
    }
    Ok(drv)
}

pub fn shutdown() {
    unsafe {
        ffi::fb_shutdown();
    }
}

pub fn device_count() -> Result<i32> {
    let mut n = 0;
    check(unsafe { ffi::fb_device_count(&mut n) }, "fb_device_count")?;
    Ok(n)
}

#[derive(Debug, Clone, Default)]
pub struct GpuInfo {
    pub name: String,
    pub device_index: i32,
    pub cc_major: i32,
    pub cc_minor: i32,
    pub driver_version: i32,
    pub runtime_version: i32,
    pub multiprocessor_count: i32,
    pub total_mem: u64,
}

pub fn gpu_info(device_index: i32) -> Result<GpuInfo> {
    let mut raw: ffi::fb_gpu_info_t = unsafe { std::mem::zeroed() };
    check(unsafe { ffi::fb_gpu_info(device_index, &mut raw) }, "fb_gpu_info")?;
    Ok(GpuInfo {
        name: cstr_to_string(&raw.name),
        device_index: raw.device_index,
        cc_major: raw.cc_major,
        cc_minor: raw.cc_minor,
        driver_version: raw.driver_version,
        runtime_version: raw.runtime_version,
        multiprocessor_count: raw.multiprocessor_count,
        total_mem: raw.total_mem,
    })
}

/// (free, total) VRAM in bytes of the current device.
pub fn mem_info() -> Result<(u64, u64)> {
    let (mut f, mut t) = (0usize, 0usize);
    check(unsafe { ffi::fb_mem_info(&mut f, &mut t) }, "fb_mem_info")?;
    Ok((f as u64, t as u64))
}

pub fn trt_set_library(path: &Path) -> Result<()> {
    let c = CString::new(path.to_string_lossy().as_bytes())
        .map_err(|_| NativeError::new(code::INVALID_ARG, "fb_trt_set_library", "path contains NUL"))?;
    check(unsafe { ffi::fb_trt_set_library(c.as_ptr()) }, "fb_trt_set_library")
}

/// TensorRT runtime version, e.g. 110400 for 11.4.0. Loads the lean runtime DLL.
pub fn trt_version() -> Result<i32> {
    let mut v = 0;
    check(unsafe { ffi::fb_trt_version(&mut v) }, "fb_trt_version")?;
    Ok(v)
}

// ---------------------------------------------------------------------------------------
// stream
// ---------------------------------------------------------------------------------------

/// A CUDA stream. `None`-like default stream is represented by passing `None` to the kernel
/// wrappers that take `Option<&Stream>`.
pub struct Stream {
    raw: *mut ffi::fb_stream,
}

// A stream handle may be moved between threads (CUDA streams are thread-safe objects).
unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Stream {
    pub fn new() -> Result<Stream> {
        let mut p = std::ptr::null_mut();
        check(unsafe { ffi::fb_stream_create(&mut p) }, "fb_stream_create")?;
        Ok(Stream { raw: p })
    }
    pub fn sync(&self) -> Result<()> {
        check(unsafe { ffi::fb_stream_sync(self.raw) }, "fb_stream_sync")
    }
    pub(crate) fn raw(&self) -> *mut ffi::fb_stream {
        self.raw
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            ffi::fb_stream_destroy(self.raw);
        }
    }
}

#[inline]
fn sraw(s: Option<&Stream>) -> *mut ffi::fb_stream {
    s.map(|s| s.raw).unwrap_or(std::ptr::null_mut())
}

/// Synchronise the default (legacy) stream.
pub fn sync_default() -> Result<()> {
    check(unsafe { ffi::fb_stream_sync(std::ptr::null_mut()) }, "fb_stream_sync")
}

// ---------------------------------------------------------------------------------------
// device memory
// ---------------------------------------------------------------------------------------

/// Owned VRAM allocation (`fb_alloc` / `fb_free`).
pub struct DeviceBuffer {
    ptr: u64,
    bytes: usize,
}

unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    pub fn alloc(bytes: usize) -> Result<DeviceBuffer> {
        let mut p: u64 = 0;
        check(unsafe { ffi::fb_alloc(bytes.max(1), &mut p) }, "fb_alloc")?;
        Ok(DeviceBuffer { ptr: p, bytes })
    }
    #[inline]
    pub fn ptr(&self) -> u64 {
        self.ptr
    }
    #[inline]
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn memset(&self, value: i32, s: Option<&Stream>) -> Result<()> {
        memset(self.ptr, value, self.bytes, s)
    }
    /// Copy `data` (host) into the buffer (synchronously for pageable memory).
    pub fn upload<T: Copy>(&self, data: &[T], s: Option<&Stream>) -> Result<()> {
        let n = std::mem::size_of_val(data);
        debug_assert!(n <= self.bytes);
        memcpy(self.ptr, data.as_ptr() as u64, n, s)
    }
    /// Copy the buffer to host memory (blocks until done: syncs the stream afterwards).
    pub fn download<T: Copy + Default>(&self, count: usize, s: Option<&Stream>) -> Result<Vec<T>> {
        let mut v = vec![T::default(); count];
        let n = count * std::mem::size_of::<T>();
        debug_assert!(n <= self.bytes);
        memcpy(v.as_mut_ptr() as u64, self.ptr, n, s)?;
        match s {
            Some(s) => s.sync()?,
            None => sync_default()?,
        }
        Ok(v)
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        unsafe {
            ffi::fb_free(self.ptr);
        }
    }
}

pub fn memcpy(dst: u64, src: u64, bytes: usize, s: Option<&Stream>) -> Result<()> {
    check(unsafe { ffi::fb_memcpy(dst, src, bytes, sraw(s)) }, "fb_memcpy")
}

pub fn memcpy2d(
    dst: u64,
    dst_pitch: usize,
    src: u64,
    src_pitch: usize,
    width_bytes: usize,
    height: usize,
    s: Option<&Stream>,
) -> Result<()> {
    check(
        unsafe { ffi::fb_memcpy2d(dst, dst_pitch, src, src_pitch, width_bytes, height, sraw(s)) },
        "fb_memcpy2d",
    )
}

pub fn memset(dst: u64, value: i32, bytes: usize, s: Option<&Stream>) -> Result<()> {
    check(unsafe { ffi::fb_memset(dst, value, bytes, sraw(s)) }, "fb_memset")
}

// ---------------------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------------------

/// A semi-planar YUV 4:2:0 picture in VRAM (as found in FFmpeg CUDA frames).
#[derive(Debug, Clone, Copy)]
pub struct YuvFrame {
    pub luma: u64,
    pub luma_pitch: usize,
    pub chroma: u64,
    pub chroma_pitch: usize,
    pub width: i32,
    pub height: i32,
    pub format: YuvFormat,
}

impl YuvFrame {
    fn raw(&self) -> ffi::fb_yuv_frame_t {
        ffi::fb_yuv_frame_t {
            luma: self.luma,
            luma_pitch: self.luma_pitch,
            chroma: self.chroma,
            chroma_pitch: self.chroma_pitch,
            width: self.width,
            height: self.height,
            format: self.format.raw(),
        }
    }
}

/// YUV -> planar RGB tensor of `dst_w x dst_h` (>= source size; edge-replicated padding).
pub fn yuv_to_rgb(
    src: &YuvFrame,
    matrix: Matrix,
    range: Range,
    dst: u64,
    dst_dtype: DType,
    dst_w: i32,
    dst_h: i32,
    s: Option<&Stream>,
) -> Result<()> {
    let raw = src.raw();
    check(
        unsafe {
            ffi::fb_yuv_to_rgb(&raw, matrix.raw(), range.raw(), dst, dst_dtype.raw(), dst_w, dst_h, sraw(s))
        },
        "fb_yuv_to_rgb",
    )
}

/// RGB tensor (`src_w x src_h`) -> YUV frame (writes `dst.width x dst.height` from the top-left).
pub fn rgb_to_yuv(
    src: u64,
    src_dtype: DType,
    src_w: i32,
    src_h: i32,
    dst: &YuvFrame,
    matrix: Matrix,
    range: Range,
    s: Option<&Stream>,
) -> Result<()> {
    let raw = dst.raw();
    check(
        unsafe {
            ffi::fb_rgb_to_yuv(src, src_dtype.raw(), src_w, src_h, &raw, matrix.raw(), range.raw(), sraw(s))
        },
        "fb_rgb_to_yuv",
    )
}

pub fn pad(src: u64, dtype: DType, w: i32, h: i32, dst: u64, dst_w: i32, dst_h: i32, s: Option<&Stream>) -> Result<()> {
    check(unsafe { ffi::fb_pad(src, dtype.raw(), w, h, dst, dst_w, dst_h, sraw(s)) }, "fb_pad")
}

#[allow(clippy::too_many_arguments)]
pub fn crop(
    src: u64,
    dtype: DType,
    src_w: i32,
    src_h: i32,
    x: i32,
    y: i32,
    dst: u64,
    dst_w: i32,
    dst_h: i32,
    s: Option<&Stream>,
) -> Result<()> {
    check(
        unsafe { ffi::fb_crop(src, dtype.raw(), src_w, src_h, x, y, dst, dst_w, dst_h, sraw(s)) },
        "fb_crop",
    )
}

pub fn convert(src: u64, src_dtype: DType, dst: u64, dst_dtype: DType, w: i32, h: i32, s: Option<&Stream>) -> Result<()> {
    check(
        unsafe { ffi::fb_convert(src, src_dtype.raw(), dst, dst_dtype.raw(), w, h, sraw(s)) },
        "fb_convert",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn tile_split(
    src: u64,
    dtype: DType,
    src_w: i32,
    src_h: i32,
    x: i32,
    y: i32,
    tile: u64,
    tile_w: i32,
    tile_h: i32,
    s: Option<&Stream>,
) -> Result<()> {
    check(
        unsafe { ffi::fb_tile_split(src, dtype.raw(), src_w, src_h, x, y, tile, tile_w, tile_h, sraw(s)) },
        "fb_tile_split",
    )
}

/// Border flags for [`tile_merge`]: which tile edges lie on the image border.
pub mod border {
    pub const LEFT: i32 = 1;
    pub const TOP: i32 = 2;
    pub const RIGHT: i32 = 4;
    pub const BOTTOM: i32 = 8;
}

#[allow(clippy::too_many_arguments)]
pub fn tile_merge(
    tile: u64,
    tile_dtype: DType,
    tile_w: i32,
    tile_h: i32,
    accum: u64,
    weight: u64,
    out_w: i32,
    out_h: i32,
    dst_x: i32,
    dst_y: i32,
    feather: i32,
    border_flags: i32,
    s: Option<&Stream>,
) -> Result<()> {
    check(
        unsafe {
            ffi::fb_tile_merge(
                tile,
                tile_dtype.raw(),
                tile_w,
                tile_h,
                accum,
                weight,
                out_w,
                out_h,
                dst_x,
                dst_y,
                feather,
                border_flags,
                sraw(s),
            )
        },
        "fb_tile_merge",
    )
}

pub fn tile_finalize(accum: u64, weight: u64, out_w: i32, out_h: i32, dst: u64, dst_dtype: DType, s: Option<&Stream>) -> Result<()> {
    check(
        unsafe { ffi::fb_tile_finalize(accum, weight, out_w, out_h, dst, dst_dtype.raw(), sraw(s)) },
        "fb_tile_finalize",
    )
}

/// Mean absolute luma difference of two tensors at 64x36, in [0,1]. Synchronises `s`.
pub fn scene_score(a: u64, b: u64, dtype: DType, w: i32, h: i32, s: Option<&Stream>) -> Result<f32> {
    let mut out = 0f32;
    check(
        unsafe { ffi::fb_scene_score(a, b, dtype.raw(), w, h, &mut out, sraw(s)) },
        "fb_scene_score",
    )?;
    Ok(out)
}

// ---------------------------------------------------------------------------------------
// engine
// ---------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TensorDesc {
    pub name: String,
    pub is_input: bool,
    pub dtype: DType,
    pub dims: Vec<i64>,
    pub min_dims: Vec<i64>,
    pub opt_dims: Vec<i64>,
    pub max_dims: Vec<i64>,
}

/// A deserialized TensorRT engine with its execution context.
/// One thread at a time (`Send`, not `Sync`).
pub struct Engine {
    raw: *mut ffi::fb_engine,
}

unsafe impl Send for Engine {}

impl Engine {
    pub fn load(path: &Path) -> Result<Engine> {
        let c = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| NativeError::new(code::INVALID_ARG, "fb_engine_load", "path contains NUL"))?;
        let mut p = std::ptr::null_mut();
        check(unsafe { ffi::fb_engine_load(c.as_ptr(), &mut p) }, "fb_engine_load")?;
        Ok(Engine { raw: p })
    }

    pub fn io_count(&self) -> Result<usize> {
        let mut n = 0;
        check(unsafe { ffi::fb_engine_io_count(self.raw, &mut n) }, "fb_engine_io_count")?;
        Ok(n as usize)
    }

    pub fn io_descs(&self) -> Result<Vec<TensorDesc>> {
        let n = self.io_count()?;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut d: ffi::fb_tensor_desc_t = unsafe { std::mem::zeroed() };
            check(unsafe { ffi::fb_engine_io_desc(self.raw, i as i32, &mut d) }, "fb_engine_io_desc")?;
            let nd = (d.nb_dims.max(0) as usize).min(8);
            out.push(TensorDesc {
                name: cstr_to_string(&d.name),
                is_input: d.is_input != 0,
                dtype: DType::from_raw(d.dtype).unwrap_or(DType::Fp32),
                dims: d.dims[..nd].to_vec(),
                min_dims: d.min_dims[..nd].to_vec(),
                opt_dims: d.opt_dims[..nd].to_vec(),
                max_dims: d.max_dims[..nd].to_vec(),
            });
        }
        Ok(out)
    }

    pub fn set_input_shape(&mut self, name: &str, dims: &[i64]) -> Result<()> {
        let c = CString::new(name).unwrap();
        check(
            unsafe { ffi::fb_engine_set_input_shape(self.raw, c.as_ptr(), dims.len() as i32, dims.as_ptr()) },
            "fb_engine_set_input_shape",
        )
    }

    pub fn io_shape(&self, name: &str) -> Result<Vec<i64>> {
        let c = CString::new(name).unwrap();
        let mut nd = 0;
        let mut dims = [0i64; 8];
        check(
            unsafe { ffi::fb_engine_io_shape(self.raw, c.as_ptr(), &mut nd, dims.as_mut_ptr()) },
            "fb_engine_io_shape",
        )?;
        Ok(dims[..(nd.max(0) as usize).min(8)].to_vec())
    }

    pub fn set_tensor(&mut self, name: &str, ptr: u64) -> Result<()> {
        let c = CString::new(name).unwrap();
        check(unsafe { ffi::fb_engine_set_tensor(self.raw, c.as_ptr(), ptr) }, "fb_engine_set_tensor")
    }

    /// Enqueue inference on `s` (asynchronous).
    pub fn run(&mut self, s: &Stream) -> Result<()> {
        check(unsafe { ffi::fb_engine_run(self.raw, s.raw) }, "fb_engine_run")
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            ffi::fb_engine_free(self.raw);
        }
    }
}
