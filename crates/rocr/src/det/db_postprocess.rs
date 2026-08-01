//! DB post-processing: threshold → connected components → minimum-area
//! rectangles → unclip → score filtering.
//!
//! Mirrors PaddleOCR's `DBPostProcess` (`boxes_from_bitmap`):
//! 1. `bitmap = prob > thresh`
//! 2. connected components (8-connectivity)
//! 3. minimum-area rectangle per component → 4 points
//! 4. filter `min_side < min_size`
//! 5. score = mean prob inside the box; filter `box_thresh > score`
//! 6. unclip with `unclip_ratio` (Clipper round join)
//! 7. re-fit a rectangle, filter `min_side < min_size + 2`

use std::collections::HashMap;

use candle_core::Tensor;

use clipper2_rust::{inflate_paths_d, EndType, JoinType, PathD, Point};

use crate::error::Error;

/// DB post-processing parameters (PP-OCRv6 small defaults).
#[derive(Debug, Clone)]
pub struct DbPostprocess {
    /// Sigmoid threshold producing the binary map.
    pub thresh: f32,
    /// Minimum mean-probability for a box.
    pub box_thresh: f32,
    pub unclip_ratio: f32,
    pub min_size: f32,
    pub max_candidates: usize,
}

impl Default for DbPostprocess {
    fn default() -> Self {
        // PP-OCRv6 small det defaults (from inference.yml).
        Self {
            thresh: 0.2,
            box_thresh: 0.45,
            unclip_ratio: 1.4,
            min_size: 3.0,
            max_candidates: 3000,
        }
    }
}

impl DbPostprocess {
    /// Extract text boxes from a `[1, 1, H, W]` probability map. Returns each
    /// box as four corner points `[x, y]` in the prob-map coordinate space.
    pub fn run(
        &self,
        prob: &Tensor,
        _device: &candle_core::Device,
    ) -> Result<Vec<Vec<[f32; 2]>>, Error> {
        let prob = prob.squeeze(0)?.squeeze(0)?; // [H, W]
        let (h, w) = prob.dims2()?;
        let data: Vec<Vec<f32>> = prob.to_vec2()?;
        let components = connected_components(h, w, &data, self.thresh);
        let mut boxes = Vec::new();
        for comp in components {
            if comp.len() < 4 {
                continue;
            }
            let hull = convex_hull(&comp);
            let rect = min_area_rect(&hull);
            let rect = order_points(rect);
            if min_side(&rect) < self.min_size {
                continue;
            }
            let score = box_score(&data, &rect);
            if self.box_thresh > score {
                continue;
            }
            let distance = polygon_area(&rect) * self.unclip_ratio / polygon_perimeter(&rect);
            let expanded = unclip(&rect, distance);
            if expanded.len() != 1 || expanded[0].len() < 4 {
                continue;
            }
            let hull2 = convex_hull_f(&expanded[0]);
            let rect2 = order_points(min_area_rect(&hull2));
            if min_side(&rect2) < self.min_size + 2.0 {
                continue;
            }
            boxes.push(rect2);
            if boxes.len() >= self.max_candidates {
                break;
            }
        }
        Ok(boxes)
    }
}

/// Union-find based connected components (8-connectivity) of `prob > thresh`.
fn connected_components(
    h: usize,
    w: usize,
    data: &[Vec<f32>],
    thresh: f32,
) -> Vec<Vec<(usize, usize)>> {
    let mut parent: Vec<usize> = (0..h * w).collect();
    fn find(parent: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while parent[r] != r {
            r = parent[r];
        }
        let mut c = x;
        while parent[c] != r {
            let n = parent[c];
            parent[c] = r;
            c = n;
        }
        r
    }
    fn union(parent: &mut [usize], a: usize, b: usize) {
        let (ra, rb) = (find(parent, a), find(parent, b));
        if ra != rb {
            parent[ra] = rb;
        }
    }
    let active = |y: usize, x: usize| data[y][x] > thresh;
    for y in 0..h {
        for x in 0..w {
            if !active(y, x) {
                continue;
            }
            for (dy, dx) in [(-1i64, -1i64), (-1, 0), (-1, 1), (0, -1)] {
                let (ny, nx) = (y as i64 + dy, x as i64 + dx);
                if ny >= 0
                    && nx >= 0
                    && (ny as usize) < h
                    && (nx as usize) < w
                    && active(ny as usize, nx as usize)
                {
                    union(&mut parent, y * w + x, ny as usize * w + nx as usize);
                }
            }
        }
    }
    let mut map: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for y in 0..h {
        for x in 0..w {
            if active(y, x) {
                let r = find(&mut parent, y * w + x);
                map.entry(r).or_default().push((x, y));
            }
        }
    }
    map.into_values().collect()
}

/// Convex hull (Andrew's monotone chain).
fn convex_hull(points: &[(usize, usize)]) -> Vec<[f32; 2]> {
    let pts: Vec<[f32; 2]> = points.iter().map(|&(x, y)| [x as f32, y as f32]).collect();
    convex_hull_f(&pts)
}

fn convex_hull_f(pts: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let mut pts = pts.to_vec();
    pts.sort_by(|a, b| {
        a[0].partial_cmp(&b[0])
            .unwrap()
            .then(a[1].partial_cmp(&b[1]).unwrap())
    });
    pts.dedup();
    if pts.len() <= 1 {
        return pts;
    }
    let cross = |o: [f32; 2], a: [f32; 2], b: [f32; 2]| {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
    let mut lower: Vec<[f32; 2]> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<[f32; 2]> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Minimum-area rectangle via rotating calipers (returns 4 corners in hull
/// coordinate order).
fn min_area_rect(hull: &[[f32; 2]]) -> [[f32; 2]; 4] {
    let n = hull.len();
    let mut min_area = f32::INFINITY;
    let mut best = [[0f32; 2]; 4];
    for i in 0..n {
        let p0 = hull[i];
        let p1 = hull[(i + 1) % n];
        let mut ex = p1[0] - p0[0];
        let mut ey = p1[1] - p0[1];
        let len = (ex * ex + ey * ey).sqrt();
        if len < 1e-9 {
            continue;
        }
        ex /= len;
        ey /= len;
        let (nx, ny) = (-ey, ex);
        let (mut min_e, mut max_e) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut min_n, mut max_n) = (f32::INFINITY, f32::NEG_INFINITY);
        for &p in hull {
            let de = (p[0] - p0[0]) * ex + (p[1] - p0[1]) * ey;
            let dn = (p[0] - p0[0]) * nx + (p[1] - p0[1]) * ny;
            min_e = min_e.min(de);
            max_e = max_e.max(de);
            min_n = min_n.min(dn);
            max_n = max_n.max(dn);
        }
        let area = (max_e - min_e) * (max_n - min_n);
        if area < min_area {
            min_area = area;
            best = [
                [
                    p0[0] + min_e * ex + min_n * nx,
                    p0[1] + min_e * ey + min_n * ny,
                ],
                [
                    p0[0] + max_e * ex + min_n * nx,
                    p0[1] + max_e * ey + min_n * ny,
                ],
                [
                    p0[0] + max_e * ex + max_n * nx,
                    p0[1] + max_e * ey + max_n * ny,
                ],
                [
                    p0[0] + min_e * ex + max_n * nx,
                    p0[1] + min_e * ey + max_n * ny,
                ],
            ];
        }
    }
    best
}

/// Order the 4 rectangle corners as `[top-left, top-right, bottom-right,
/// bottom-left]` (matches PaddleOCR's `get_mini_boxes`).
fn order_points(rect: [[f32; 2]; 4]) -> Vec<[f32; 2]> {
    let mut pts = rect.to_vec();
    pts.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
    let (i1, i4) = if pts[1][1] > pts[0][1] {
        (0, 1)
    } else {
        (1, 0)
    };
    let (i2, i3) = if pts[3][1] > pts[2][1] {
        (2, 3)
    } else {
        (3, 2)
    };
    vec![pts[i1], pts[i2], pts[i3], pts[i4]]
}

fn min_side(poly: &[[f32; 2]]) -> f32 {
    let n = poly.len();
    let mut min = f32::INFINITY;
    for i in 0..n {
        let j = (i + 1) % n;
        let d = ((poly[i][0] - poly[j][0]).powi(2) + (poly[i][1] - poly[j][1]).powi(2)).sqrt();
        min = min.min(d);
    }
    min
}

fn polygon_area(poly: &[[f32; 2]]) -> f32 {
    let n = poly.len();
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1];
    }
    area.abs() / 2.0
}

fn polygon_perimeter(poly: &[[f32; 2]]) -> f32 {
    let n = poly.len();
    let mut per = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        per += ((poly[i][0] - poly[j][0]).powi(2) + (poly[i][1] - poly[j][1]).powi(2)).sqrt();
    }
    per
}

/// Polygon offset (unclip) via Clipper2 with round joins (pyclipper equivalent).
fn unclip(poly: &[[f32; 2]], distance: f32) -> Vec<Vec<[f32; 2]>> {
    if distance <= 0.0 {
        return vec![poly.to_vec()];
    }
    let path: PathD = poly
        .iter()
        .map(|p| Point {
            x: p[0] as f64,
            y: p[1] as f64,
        })
        .collect();
    let paths = inflate_paths_d(
        &vec![path],
        distance as f64,
        JoinType::Round,
        EndType::Polygon,
        2.0,
        2,
        0.25,
    );
    paths
        .iter()
        .map(|p| p.iter().map(|pt| [pt.x as f32, pt.y as f32]).collect())
        .collect()
}

fn point_in_poly(px: f32, py: f32, poly: &[[f32; 2]]) -> bool {
    let n = poly.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        if (poly[i][1] > py) != (poly[j][1] > py) {
            let xint = (poly[j][0] - poly[i][0]) * (py - poly[i][1]) / (poly[j][1] - poly[i][1])
                + poly[i][0];
            if px < xint {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// Mean probability inside a polygon (matches PaddleOCR `box_score_fast`).
fn box_score(data: &[Vec<f32>], poly: &[[f32; 2]]) -> f32 {
    let h = data.len();
    let w = data[0].len();
    let (mut xmin, mut xmax) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut ymin, mut ymax) = (f32::INFINITY, f32::NEG_INFINITY);
    for p in poly {
        xmin = xmin.min(p[0]);
        xmax = xmax.max(p[0]);
        ymin = ymin.min(p[1]);
        ymax = ymax.max(p[1]);
    }
    let x0 = (xmin.floor() as i64).clamp(0, w as i64 - 1);
    let x1 = (xmax.ceil() as i64).clamp(0, w as i64 - 1);
    let y0 = (ymin.floor() as i64).clamp(0, h as i64 - 1);
    let y1 = (ymax.ceil() as i64).clamp(0, h as i64 - 1);
    let mut sum = 0.0f32;
    let mut cnt = 0usize;
    for y in y0..=y1 {
        for x in x0..=x1 {
            if point_in_poly(x as f32 + 0.5, y as f32 + 0.5, poly) {
                sum += data[y as usize][x as usize];
                cnt += 1;
            }
        }
    }
    if cnt == 0 {
        0.0
    } else {
        sum / cnt as f32
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::*;

    fn prob(
        h: usize,
        w: usize,
        regions: &[(usize, usize, usize, usize)],
        fill: f32,
    ) -> Vec<Vec<f32>> {
        let mut m = vec![vec![0.1f32; w]; h];
        for &(y0, y1, x0, x1) in regions {
            for row in &mut m[y0..y1] {
                for cell in &mut row[x0..x1] {
                    *cell = fill;
                }
            }
        }
        m
    }

    #[test]
    fn connected_components_finds_separate_regions() {
        // Two isolated 1px seeds at (1,1) and (4,4).
        let mut data = vec![vec![0.0f32; 6]; 6];
        data[1][1] = 1.0;
        data[4][4] = 1.0;
        let comps = connected_components(6, 6, &data, 0.5);
        assert_eq!(comps.len(), 2, "expected two components");
        let sizes: Vec<_> = comps.iter().map(|c| c.len()).collect();
        assert!(sizes.contains(&1) && sizes.contains(&1));
    }

    #[test]
    fn connected_components_8_connectivity_diagonal() {
        let mut data = vec![vec![0.0f32; 3]; 3];
        data[0][0] = 1.0;
        data[1][1] = 1.0; // diagonal neighbor via 8-connectivity
        let comps = connected_components(3, 3, &data, 0.5);
        assert_eq!(comps.len(), 1, "diagonal should merge into one component");
        assert_eq!(comps[0].len(), 2);
    }

    #[test]
    fn connected_components_ignores_below_threshold() {
        let mut data = vec![vec![0.0f32; 3]; 3];
        data[1][1] = 0.2; // below thresh 0.5
        assert!(connected_components(3, 3, &data, 0.5).is_empty());
    }

    #[test]
    fn convex_hull_of_square_is_corners() {
        let pts = vec![(0usize, 0usize), (0, 4), (4, 0), (4, 4), (2, 2), (1, 3)];
        let hull = convex_hull(&pts);
        assert_eq!(hull.len(), 4, "hull: {hull:?}");
        let area = polygon_area(&hull);
        assert!((area - 16.0).abs() < 1e-3, "area {area}");
    }

    #[test]
    fn convex_hull_degenerate_points() {
        let pts = vec![(2usize, 2usize)];
        assert_eq!(convex_hull(&pts).len(), 1);
        let pts = vec![(0usize, 0usize), (1usize, 1usize)];
        assert_eq!(convex_hull(&pts).len(), 2);
    }

    #[test]
    fn order_points_returns_tl_tr_br_bl() {
        let rect = [[4.0, 4.0], [0.0, 4.0], [0.0, 0.0], [4.0, 0.0]];
        let ordered = order_points(rect);
        assert_eq!(ordered[0], [0.0, 0.0]); // TL
        assert_eq!(ordered[1], [4.0, 0.0]); // TR
        assert_eq!(ordered[2], [4.0, 4.0]); // BR
        assert_eq!(ordered[3], [0.0, 4.0]); // BL
    }

    #[test]
    fn polygon_measurements_known_rectangle() {
        let rect = [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]];
        assert!((polygon_area(&rect) - 12.0).abs() < 1e-4);
        assert!((polygon_perimeter(&rect) - 14.0).abs() < 1e-4);
        assert!((min_side(&rect) - 3.0).abs() < 1e-4);
    }

    #[test]
    fn point_in_poly_inside_outside() {
        let tri = [[0.0, 0.0], [4.0, 0.0], [2.0, 4.0]];
        assert!(point_in_poly(2.0, 1.0, &tri));
        assert!(!point_in_poly(0.5, 3.0, &tri));
    }

    #[test]
    fn box_score_means_probability_inside() {
        let data = prob(8, 8, &[(2, 6, 2, 6)], 0.9);
        let rect = [[2.0, 2.0], [6.0, 2.0], [6.0, 6.0], [2.0, 6.0]];
        let score = box_score(&data, &rect);
        assert!((score - 0.9).abs() < 1e-4, "score {score}");
        let empty = vec![vec![0.0f32; 8]; 8];
        assert_eq!(box_score(&empty, &rect), 0.0);
    }

    #[test]
    fn run_detects_single_rect() {
        let m = prob(12, 12, &[(3, 9, 4, 10)], 0.9);
        let t = Tensor::from_vec(m.concat(), (12, 12), &Device::Cpu).unwrap();
        let t = t.unsqueeze(0).unwrap().unsqueeze(0).unwrap();
        let boxes = DbPostprocess::default().run(&t, &Device::Cpu).unwrap();
        assert_eq!(boxes.len(), 1, "expected exactly one box: {boxes:?}");
        // The box should be near the [4..10] x [3..9] region.
        let min_x = boxes[0].iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
        let max_x = boxes[0]
            .iter()
            .map(|p| p[0])
            .fold(f32::NEG_INFINITY, f32::max);
        let min_y = boxes[0].iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
        let max_y = boxes[0]
            .iter()
            .map(|p| p[1])
            .fold(f32::NEG_INFINITY, f32::max);
        assert!((0.0..=5.0).contains(&min_x), "min_x {min_x}");
        assert!((9.0..=11.0).contains(&max_x), "max_x {max_x}");
        assert!((0.0..=4.0).contains(&min_y), "min_y {min_y}");
        assert!((8.0..=10.0).contains(&max_y), "max_y {max_y}");
    }

    #[test]
    fn run_empty_map_no_boxes() {
        let m = vec![vec![0.1f32; 12]; 12];
        let t = Tensor::from_vec(m.concat(), (12, 12), &Device::Cpu).unwrap();
        let t = t.unsqueeze(0).unwrap().unsqueeze(0).unwrap();
        assert!(DbPostprocess::default()
            .run(&t, &Device::Cpu)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn run_filters_weak_boxes() {
        // Probability below box_thresh (0.45) but above thresh (0.2): no box.
        let m = prob(12, 12, &[(3, 9, 4, 10)], 0.3);
        let t = Tensor::from_vec(m.concat(), (12, 12), &Device::Cpu).unwrap();
        let t = t.unsqueeze(0).unwrap().unsqueeze(0).unwrap();
        let boxes = DbPostprocess::default().run(&t, &Device::Cpu).unwrap();
        // 0.3 < box_thresh → dropped.
        assert!(boxes.is_empty(), "weak box should be filtered: {boxes:?}");
    }
}
