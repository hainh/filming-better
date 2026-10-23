//! Timestamp math: VFR -> CFR grid and output pts numbering.

use crate::io::{AVRational, NOPTS};

/// Slot index (frame number on the CFR grid) of a timestamp offset `dpts` (stream time base
/// units since the first frame), for a grid of `fps`.
pub fn slot_of(dpts: i64, tb: AVRational, fps: AVRational) -> i64 {
    // dpts * tb.num/tb.den seconds * fps.num/fps.den frames/s, rounded to nearest
    let num = dpts as i128 * tb.num as i128 * fps.num as i128;
    let den = tb.den as i128 * fps.den as i128;
    if den == 0 {
        return 0;
    }
    // floor((2*num + den) / (2*den)) = round half up
    (2 * num + den).div_euclid(2 * den) as i64
}

/// Output timestamp (in ticks of `1/out_fps`) of the first frame, i.e. where the input's
/// video started: `round(pts0 * tb * out_fps)`.
pub fn start_ticks(pts0: i64, tb: AVRational, out_fps: AVRational) -> i64 {
    slot_of(pts0, tb, out_fps)
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Slot {
    /// Another frame already occupies this grid slot: drop this one.
    Drop,
    /// Emit this frame, preceded by `dups` repeats of the previous frame (gap filling).
    Emit { dups: u32 },
}

/// Maps decoded frames (arbitrary pts) onto a constant-frame-rate grid: gaps are filled by
/// repeating the previous frame, surplus frames in one slot are dropped.
#[derive(Debug)]
pub struct CfrGrid {
    tb: AVRational,
    fps: AVRational,
    pts0: Option<i64>,
    next_slot: i64,
    max_gap: i64,
    pub dropped: u64,
    pub duplicated: u64,
}

impl CfrGrid {
    pub fn new(tb: AVRational, fps: AVRational) -> CfrGrid {
        let fpsf = fps.num as f64 / fps.den.max(1) as f64;
        CfrGrid {
            tb,
            fps,
            pts0: None,
            next_slot: 0,
            max_gap: (fpsf * 600.0).ceil() as i64,
            dropped: 0,
            duplicated: 0,
        }
    }

    pub fn first_pts(&self) -> Option<i64> {
        self.pts0
    }

    /// Number of grid slots used so far.
    pub fn slots(&self) -> i64 {
        self.next_slot
    }

    pub fn push(&mut self, pts: i64) -> Slot {
        let slot = if pts == NOPTS {
            self.next_slot
        } else {
            let p0 = *self.pts0.get_or_insert(pts);
            slot_of(pts - p0, self.tb, self.fps)
        };
        if slot < self.next_slot {
            self.dropped += 1;
            return Slot::Drop;
        }
        let mut gap = slot - self.next_slot;
        if gap > self.max_gap {
            tracing::warn!("timestamp jump of {gap} frames clamped to {}", self.max_gap);
            gap = self.max_gap;
        }
        self.next_slot += gap + 1;
        self.duplicated += gap as u64;
        Slot::Emit { dups: gap as u32 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(n: i32, d: i32) -> AVRational {
        AVRational { num: n, den: d }
    }

    #[test]
    fn slots_cfr_30() {
        let tb = q(1, 15360);
        let fps = q(30, 1);
        for i in 0..10 {
            assert_eq!(slot_of(i * 512, tb, fps), i);
        }
        // jitter within half a frame still maps to the same slot
        assert_eq!(slot_of(512 + 200, tb, fps), 1);
        assert_eq!(slot_of(512 - 200, tb, fps), 1);
    }

    #[test]
    fn slots_ntsc() {
        let tb = q(1, 90000);
        let fps = q(30000, 1001);
        assert_eq!(slot_of(3003, tb, fps), 1);
        assert_eq!(slot_of(3003 * 100, tb, fps), 100);
    }

    #[test]
    fn cfr_passthrough() {
        let mut g = CfrGrid::new(q(1, 30), q(30, 1));
        for i in 0..5 {
            assert_eq!(g.push(100 + i), Slot::Emit { dups: 0 });
        }
        assert_eq!(g.slots(), 5);
        assert_eq!(g.first_pts(), Some(100));
    }

    #[test]
    fn cfr_vfr_gap_and_drop() {
        // 30fps grid, timebase 1/1000: frames at 0, 33, 100 (gap of 2 slots), 105 (drop)
        let mut g = CfrGrid::new(q(1, 1000), q(30, 1));
        assert_eq!(g.push(0), Slot::Emit { dups: 0 });
        assert_eq!(g.push(33), Slot::Emit { dups: 0 });
        assert_eq!(g.push(100), Slot::Emit { dups: 1 }); // slot 3: slot 2 filled by a repeat
        assert_eq!(g.push(105), Slot::Drop); // still slot 3
        assert_eq!(g.slots(), 4);
        assert_eq!(g.dropped, 1);
        assert_eq!(g.duplicated, 1);
    }

    #[test]
    fn cfr_missing_pts() {
        let mut g = CfrGrid::new(q(1, 30), q(30, 1));
        assert_eq!(g.push(NOPTS), Slot::Emit { dups: 0 });
        assert_eq!(g.push(NOPTS), Slot::Emit { dups: 0 });
        assert_eq!(g.slots(), 2);
    }

    #[test]
    fn start_offset_in_output_ticks() {
        // video starts at 0.5s; output 60 fps -> tick 30
        assert_eq!(start_ticks(15, q(1, 30), q(60, 1)), 30);
        assert_eq!(start_ticks(0, q(1, 30), q(60, 1)), 0);
    }

    #[test]
    fn output_count_keeps_duration() {
        // N frames in at fps, interp x2 -> 2N frames at 2*fps keep the duration
        let n_in: i64 = 214;
        let fps_in = 30.0;
        let n_out = n_in * 2;
        assert!((n_out as f64 / (fps_in * 2.0) - n_in as f64 / fps_in).abs() < 1e-9);
    }
}
