//! VRAM buffer pool + the tensor type that flows between pipeline stages.

use crate::native::{self, DType, DeviceBuffer};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Recycles device allocations by size so steady-state processing does no `cudaMalloc`.
pub struct Pool {
    free: Mutex<HashMap<usize, Vec<DeviceBuffer>>>,
}

impl Pool {
    pub fn new() -> Arc<Pool> {
        Arc::new(Pool { free: Mutex::new(HashMap::new()) })
    }

    pub fn get(self: &Arc<Self>, bytes: usize) -> native::Result<PooledBuf> {
        let reused = self.free.lock().unwrap().get_mut(&bytes).and_then(|v| v.pop());
        let buf = match reused {
            Some(b) => b,
            None => match DeviceBuffer::alloc(bytes) {
                Ok(b) => b,
                Err(e) if e.is_oom() => {
                    // give back everything we cached and retry once
                    self.free.lock().unwrap().clear();
                    DeviceBuffer::alloc(bytes)?
                }
                Err(e) => return Err(e),
            },
        };
        Ok(PooledBuf { buf: Some(buf), pool: self.clone() })
    }

    /// Drop all cached (currently unused) buffers.
    pub fn trim(&self) {
        self.free.lock().unwrap().clear();
    }

    pub fn cached_bytes(&self) -> usize {
        self.free.lock().unwrap().iter().map(|(k, v)| k * v.len()).sum()
    }
}

/// A pool allocation that returns to its pool when dropped.
pub struct PooledBuf {
    buf: Option<DeviceBuffer>,
    pool: Arc<Pool>,
}

impl PooledBuf {
    #[inline]
    pub fn ptr(&self) -> u64 {
        self.buf.as_ref().unwrap().ptr()
    }
    #[inline]
    pub fn bytes(&self) -> usize {
        self.buf.as_ref().unwrap().bytes()
    }
}

impl Drop for PooledBuf {
    fn drop(&mut self) {
        if let Some(b) = self.buf.take() {
            self.pool.free.lock().unwrap().entry(b.bytes()).or_default().push(b);
        }
    }
}

/// Planar RGB (CHW, batch 1) tensor in VRAM. The allocation holds `pw x ph` per plane; the
/// picture content occupies the top-left `w x h` (the rest is edge-replicated padding).
pub struct Tensor {
    pub buf: PooledBuf,
    pub w: i32,
    pub h: i32,
    pub pw: i32,
    pub ph: i32,
    pub dtype: DType,
}

impl Tensor {
    pub fn alloc(pool: &Arc<Pool>, w: i32, h: i32, pw: i32, ph: i32, dtype: DType) -> native::Result<Tensor> {
        debug_assert!(pw >= w && ph >= h);
        let bytes = 3 * pw as usize * ph as usize * dtype.bytes();
        Ok(Tensor { buf: pool.get(bytes)?, w, h, pw, ph, dtype })
    }
    /// Tensor with no padding.
    pub fn alloc_exact(pool: &Arc<Pool>, w: i32, h: i32, dtype: DType) -> native::Result<Tensor> {
        Tensor::alloc(pool, w, h, w, h, dtype)
    }
    #[inline]
    pub fn ptr(&self) -> u64 {
        self.buf.ptr()
    }
    #[inline]
    pub fn is_exact(&self) -> bool {
        self.pw == self.w && self.ph == self.h
    }
}

/// Grow-only scratch allocations (not pooled; sized once per run).
#[derive(Default)]
pub struct Scratch {
    bufs: Vec<Option<DeviceBuffer>>,
}

impl Scratch {
    pub fn get(&mut self, slot: usize, bytes: usize) -> native::Result<u64> {
        if self.bufs.len() <= slot {
            self.bufs.resize_with(slot + 1, || None);
        }
        let need = match &self.bufs[slot] {
            Some(b) => b.bytes() < bytes,
            None => true,
        };
        if need {
            self.bufs[slot] = None; // free before re-allocating
            self.bufs[slot] = Some(DeviceBuffer::alloc(bytes)?);
        }
        Ok(self.bufs[slot].as_ref().unwrap().ptr())
    }
}

/// IEEE binary32 -> binary16 bits (round to nearest even; handles the normal range).
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        // subnormal half: truncate (precision is irrelevant at this magnitude)
        let m = (man | 0x80_0000) >> (14 - e);
        return sign | m as u16;
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) != 0) {
        h += 1;
    }
    sign | h as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_conversion() {
        assert_eq!(f32_to_f16_bits(0.0), 0);
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(0.5), 0x3800);
        assert_eq!(f32_to_f16_bits(-2.0), 0xc000);
        // 1/3 -> 0x3555 (0.33325...)
        assert_eq!(f32_to_f16_bits(1.0 / 3.0), 0x3555);
        // 2/3 -> 0x3955
        assert_eq!(f32_to_f16_bits(2.0 / 3.0), 0x3955);
        assert_eq!(f32_to_f16_bits(0.25), 0x3400);
        assert_eq!(f32_to_f16_bits(0.75), 0x3a00);
        assert_eq!(f32_to_f16_bits(1e9), 0x7c00);
    }
}
