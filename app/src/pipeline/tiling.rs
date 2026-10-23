//! Tile layout math for the upscaler.

use crate::native::border;

pub fn round_up(v: i32, m: i32) -> i32 {
    let m = m.max(1);
    (v + m - 1) / m * m
}

pub fn round_down(v: i32, m: i32) -> i32 {
    let m = m.max(1);
    v / m * m
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AxisLayout {
    /// Tile length along this axis (multiple of `pm`, >= `min_tile`).
    pub tile: i32,
    /// Tile start positions (strictly increasing; the last tile is aligned to the end).
    pub pos: Vec<i32>,
    /// Overlap actually used between neighbouring tiles.
    pub overlap: i32,
}

/// Split an axis of length `len` into tiles of at most `max_tile` with `overlap` px overlap.
/// When `len <= max_tile` there is one tile (padded up to `pm` / `min_tile`).
pub fn axis_layout(len: i32, max_tile: i32, overlap: i32, pm: i32, min_tile: i32) -> AxisLayout {
    let pm = pm.max(1);
    let min_tile = round_up(min_tile.max(1), pm);
    let max_tile = round_down(max_tile, pm).max(min_tile);
    if len <= max_tile {
        let tile = round_up(len, pm).max(min_tile);
        return AxisLayout { tile, pos: vec![0], overlap: 0 };
    }
    let tile = max_tile;
    let ov = overlap.clamp(0, tile / 4);
    let stride = tile - ov;
    let n = (len - tile + stride - 1) / stride + 1;
    let mut pos: Vec<i32> = (0..n - 1).map(|i| i * stride).collect();
    pos.push(len - tile);
    AxisLayout { tile, pos, overlap: ov }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileRect {
    pub x: i32,
    pub y: i32,
    /// `native::border::*` flags of tile edges lying on the image border.
    pub border: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileLayout {
    pub tile_w: i32,
    pub tile_h: i32,
    pub overlap: i32,
    pub tiles: Vec<TileRect>,
}

impl TileLayout {
    pub fn is_single(&self) -> bool {
        self.tiles.len() == 1
    }
}

/// Layout for a `w x h` image with tiles bounded by `max_w x max_h`.
#[allow(clippy::too_many_arguments)]
pub fn layout(w: i32, h: i32, max_w: i32, max_h: i32, overlap: i32, pm: i32, min_w: i32, min_h: i32) -> TileLayout {
    let ax = axis_layout(w, max_w, overlap, pm, min_w);
    let ay = axis_layout(h, max_h, overlap, pm, min_h);
    let mut tiles = Vec::with_capacity(ax.pos.len() * ay.pos.len());
    for &y in &ay.pos {
        for &x in &ax.pos {
            let mut b = 0;
            if x == 0 {
                b |= border::LEFT;
            }
            if y == 0 {
                b |= border::TOP;
            }
            if x + ax.tile >= w {
                b |= border::RIGHT;
            }
            if y + ay.tile >= h {
                b |= border::BOTTOM;
            }
            tiles.push(TileRect { x, y, border: b });
        }
    }
    TileLayout { tile_w: ax.tile, tile_h: ay.tile, overlap: ax.overlap.max(ay.overlap), tiles }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_tile_fits() {
        let l = layout(576, 1024, 1920, 1088, 32, 8, 64, 64);
        assert!(l.is_single());
        assert_eq!((l.tile_w, l.tile_h), (576, 1024));
        assert_eq!(l.tiles[0].border, 15);
    }

    #[test]
    fn single_tile_padded_to_multiple_and_min() {
        let l = layout(30, 100, 1920, 1088, 32, 8, 64, 64);
        assert_eq!((l.tile_w, l.tile_h), (64, 104));
    }

    #[test]
    fn multi_tile_covers_everything() {
        // 4K frame, tiles of at most 1920x1088, overlap 32
        let l = layout(3840, 2160, 1920, 1088, 32, 8, 64, 64);
        assert_eq!((l.tile_w, l.tile_h), (1920, 1088));
        let xs: Vec<i32> = l.tiles.iter().take(3).map(|t| t.x).collect();
        assert_eq!(xs, vec![0, 1888, 1920]);
        let mut cov_x = vec![false; 3840];
        let mut cov_y = vec![false; 2160];
        for t in &l.tiles {
            for x in t.x..t.x + l.tile_w {
                cov_x[x as usize] = true;
            }
            for y in t.y..t.y + l.tile_h {
                cov_y[y as usize] = true;
            }
            assert!(t.x + l.tile_w <= 3840 && t.y + l.tile_h <= 2160);
        }
        assert!(cov_x.iter().all(|&c| c) && cov_y.iter().all(|&c| c));
    }

    #[test]
    fn overlap_at_least_requested() {
        let a = axis_layout(1000, 512, 32, 8, 64);
        for w in a.pos.windows(2) {
            assert!(w[0] + a.tile - w[1] >= 32, "{:?}", a);
            assert!(w[1] > w[0]);
        }
        assert_eq!(*a.pos.last().unwrap() + a.tile, 1000);
    }

    #[test]
    fn border_flags() {
        let l = layout(1000, 500, 512, 512, 32, 1, 64, 64);
        let first = l.tiles.first().unwrap();
        assert_eq!(first.border & (border::LEFT | border::TOP), border::LEFT | border::TOP);
        assert_eq!(first.border & border::RIGHT, 0);
        let last = l.tiles.last().unwrap();
        assert_ne!(last.border & border::RIGHT, 0);
        assert_ne!(last.border & border::BOTTOM, 0);
    }

    #[test]
    fn rounding() {
        assert_eq!(round_up(1081, 64), 1088);
        assert_eq!(round_up(1088, 64), 1088);
        assert_eq!(round_down(1919, 8), 1912);
    }
}
