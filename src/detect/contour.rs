//! Pure Rust contour extraction and minimum-area rotated rectangle computation.
//!
//! Replaces OpenCV's `findContours` + `minAreaRect` with lightweight
//! algorithms suitable for DBNet post-processing of binary probability maps.

/// A 2D point (floating point).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Point2f {
    pub x: f32,
    pub y: f32,
}

/// A 2D point (integer, for bitmap operations).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Point2i {
    pub x: i32,
    pub y: i32,
}

/// A rotated rectangle defined by center, size, and angle.
#[derive(Debug, Clone)]
pub(crate) struct RotatedRect {
    pub center: Point2f,
    pub width: f32,
    pub height: f32,
    /// Rotation angle in radians.
    pub angle: f32,
}

impl RotatedRect {
    /// Get the 4 corner points of the rotated rectangle.
    /// Returns points in order: TopLeft, TopRight, BottomRight, BottomLeft.
    pub fn corners(&self) -> [[f32; 2]; 4] {
        let cos_a = self.angle.cos();
        let sin_a = self.angle.sin();
        let hw = self.width / 2.0;
        let hh = self.height / 2.0;

        // Corner offsets relative to center (before rotation)
        let offsets = [
            (-hw, -hh), // top-left
            (hw, -hh),  // top-right
            (hw, hh),   // bottom-right
            (-hw, hh),  // bottom-left
        ];

        let mut corners = [[0.0f32; 2]; 4];
        for (i, (dx, dy)) in offsets.iter().enumerate() {
            corners[i][0] = self.center.x + dx * cos_a - dy * sin_a;
            corners[i][1] = self.center.y + dx * sin_a + dy * cos_a;
        }
        corners
    }

    /// Calculate the area of the rotated rectangle.
    pub fn area(&self) -> f32 {
        self.width * self.height
    }
}

/// Extract external contours from a binary bitmap using Moore neighborhood tracing.
///
/// The bitmap is represented as a flat `u8` slice where non-zero values are foreground.
///
/// Returns a list of contours, where each contour is a list of (x, y) integer points.
pub(crate) fn find_contours(bitmap: &[u8], width: usize, height: usize) -> Vec<Vec<Point2i>> {
    let mut visited = vec![false; width * height];
    let mut contours = Vec::new();

    // 8-connected neighborhood directions (clockwise from right)
    let dirs: [(i32, i32); 8] = [
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    ];

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if bitmap[idx] == 0 || visited[idx] {
                continue;
            }

            // Check if this is a border pixel (has at least one background neighbor or is on edge)
            if !is_border_pixel(bitmap, width, height, x, y) {
                continue;
            }

            // Trace contour using Moore neighborhood algorithm
            let contour = trace_contour(bitmap, &mut visited, width, height, x, y, &dirs);
            if contour.len() >= 3 {
                contours.push(contour);
            }
        }
    }

    contours
}

/// Check if a pixel is on the border of a foreground region.
fn is_border_pixel(bitmap: &[u8], width: usize, height: usize, x: usize, y: usize) -> bool {
    if x == 0 || y == 0 || x == width - 1 || y == height - 1 {
        return true;
    }

    // Check 4-connected neighbors
    let neighbors = [(0isize, -1isize), (0, 1), (-1, 0), (1, 0)];
    for (dx, dy) in &neighbors {
        let nx = (x as isize + dx) as usize;
        let ny = (y as isize + dy) as usize;
        if bitmap[ny * width + nx] == 0 {
            return true;
        }
    }
    false
}

/// Trace a single contour starting from (start_x, start_y) using Moore neighborhood.
fn trace_contour(
    bitmap: &[u8],
    visited: &mut [bool],
    width: usize,
    height: usize,
    start_x: usize,
    start_y: usize,
    dirs: &[(i32, i32); 8],
) -> Vec<Point2i> {
    let mut contour = Vec::new();
    let mut x = start_x as i32;
    let mut y = start_y as i32;
    let mut dir = 0usize; // Start searching from direction 0 (right)

    let w = width as i32;
    let h = height as i32;

    loop {
        contour.push(Point2i { x, y });
        if (y as usize) < height && (x as usize) < width {
            visited[y as usize * width + x as usize] = true;
        }

        // Search for next border pixel in Moore neighborhood
        let mut found = false;
        let search_start = (dir + 5) % 8; // Start from dir-3 (backtrack direction + 1)

        for i in 0..8 {
            let d = (search_start + i) % 8;
            let nx = x + dirs[d].0;
            let ny = y + dirs[d].1;

            if nx < 0 || ny < 0 || nx >= w || ny >= h {
                continue;
            }

            if bitmap[ny as usize * width + nx as usize] != 0 {
                if nx == start_x as i32 && ny == start_y as i32 && contour.len() > 2 {
                    // Returned to start — contour complete
                    return contour;
                }
                x = nx;
                y = ny;
                dir = d;
                found = true;
                break;
            }
        }

        if !found || contour.len() > width * height {
            // Isolated pixel or safety cutoff
            break;
        }
    }

    contour
}

/// Simplify a contour using the Douglas-Peucker algorithm.
///
/// Reduces the number of vertices while preserving the overall shape.
pub(crate) fn simplify_contour(contour: &[Point2i], epsilon: f32) -> Vec<Point2i> {
    if contour.len() <= 2 {
        return contour.to_vec();
    }

    // Find the point with maximum distance from the line (first, last)
    let first = contour[0];
    let last = contour[contour.len() - 1];

    let mut max_dist = 0.0f32;
    let mut max_idx = 0;

    for (i, p) in contour.iter().enumerate().skip(1).take(contour.len() - 2) {
        let dist = point_to_line_distance(p, &first, &last);
        if dist > max_dist {
            max_dist = dist;
            max_idx = i;
        }
    }

    if max_dist > epsilon {
        // Recursively simplify both halves
        let mut left = simplify_contour(&contour[..=max_idx], epsilon);
        let right = simplify_contour(&contour[max_idx..], epsilon);
        left.pop(); // Remove duplicate point at the junction
        left.extend(right);
        left
    } else {
        vec![first, last]
    }
}

/// Perpendicular distance from point `p` to the line defined by `a` and `b`.
fn point_to_line_distance(p: &Point2i, a: &Point2i, b: &Point2i) -> f32 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len_sq = (dx * dx + dy * dy) as f32;

    if len_sq < 1e-10 {
        // a and b are the same point
        let px = p.x - a.x;
        let py = p.y - a.y;
        return ((px * px + py * py) as f32).sqrt();
    }

    let numerator = ((dy as f32) * (p.x - a.x) as f32 - (dx as f32) * (p.y - a.y) as f32).abs();
    numerator / len_sq.sqrt()
}

/// Compute the convex hull of a set of points using Andrew's monotone chain algorithm.
///
/// Returns points in counter-clockwise order.
pub(crate) fn convex_hull(points: &[Point2i]) -> Vec<Point2i> {
    if points.len() <= 3 {
        return points.to_vec();
    }

    let mut sorted: Vec<Point2i> = points.to_vec();
    sorted.sort_by(|a, b| a.x.cmp(&b.x).then(a.y.cmp(&b.y)));
    sorted.dedup();

    if sorted.len() <= 2 {
        return sorted;
    }

    let n = sorted.len();
    let mut hull = Vec::with_capacity(2 * n);

    // Build lower hull
    for &p in &sorted {
        while hull.len() >= 2 && cross(&hull[hull.len() - 2], &hull[hull.len() - 1], &p) <= 0 {
            hull.pop();
        }
        hull.push(p);
    }

    // Build upper hull
    let lower_len = hull.len() + 1;
    for &p in sorted.iter().rev().skip(1) {
        while hull.len() >= lower_len && cross(&hull[hull.len() - 2], &hull[hull.len() - 1], &p) <= 0
        {
            hull.pop();
        }
        hull.push(p);
    }

    hull.pop(); // Remove the duplicate last point
    hull
}

/// Cross product of vectors OA and OB where O = `o`, A = `a`, B = `b`.
#[inline]
fn cross(o: &Point2i, a: &Point2i, b: &Point2i) -> i64 {
    (a.x as i64 - o.x as i64) * (b.y as i64 - o.y as i64)
        - (a.y as i64 - o.y as i64) * (b.x as i64 - o.x as i64)
}

/// Compute the minimum-area rotated bounding rectangle for a set of points.
///
/// Uses the rotating calipers algorithm on the convex hull.
pub(crate) fn min_area_rect(points: &[Point2i]) -> RotatedRect {
    if points.is_empty() {
        return RotatedRect {
            center: Point2f { x: 0.0, y: 0.0 },
            width: 0.0,
            height: 0.0,
            angle: 0.0,
        };
    }

    if points.len() == 1 {
        return RotatedRect {
            center: Point2f {
                x: points[0].x as f32,
                y: points[0].y as f32,
            },
            width: 0.0,
            height: 0.0,
            angle: 0.0,
        };
    }

    if points.len() == 2 {
        let dx = (points[1].x - points[0].x) as f32;
        let dy = (points[1].y - points[0].y) as f32;
        let length = (dx * dx + dy * dy).sqrt();
        let angle = dy.atan2(dx);
        return RotatedRect {
            center: Point2f {
                x: (points[0].x + points[1].x) as f32 / 2.0,
                y: (points[0].y + points[1].y) as f32 / 2.0,
            },
            width: length,
            height: 0.0,
            angle,
        };
    }

    let hull = convex_hull(points);
    if hull.len() < 3 {
        // Degenerate case — fallback to simple bounding box
        return axis_aligned_bounding_rect(points);
    }

    rotating_calipers_min_rect(&hull)
}

/// Rotating calipers algorithm to find the minimum area bounding rectangle.
fn rotating_calipers_min_rect(hull: &[Point2i]) -> RotatedRect {
    let n = hull.len();
    let mut min_area = f32::MAX;
    let mut best = RotatedRect {
        center: Point2f { x: 0.0, y: 0.0 },
        width: 0.0,
        height: 0.0,
        angle: 0.0,
    };

    for i in 0..n {
        let j = (i + 1) % n;
        let edge_x = (hull[j].x - hull[i].x) as f32;
        let edge_y = (hull[j].y - hull[i].y) as f32;
        let edge_len = (edge_x * edge_x + edge_y * edge_y).sqrt();

        if edge_len < 1e-10 {
            continue;
        }

        // Unit vector along this edge
        let ux = edge_x / edge_len;
        let uy = edge_y / edge_len;

        // Project all hull points onto the edge direction and its perpendicular
        let mut min_proj = f32::MAX;
        let mut max_proj = f32::NEG_INFINITY;
        let mut min_perp = f32::MAX;
        let mut max_perp = f32::NEG_INFINITY;

        for p in hull {
            let dx = p.x as f32 - hull[i].x as f32;
            let dy = p.y as f32 - hull[i].y as f32;

            let proj = dx * ux + dy * uy; // projection along edge
            let perp = -dx * uy + dy * ux; // projection perpendicular to edge

            min_proj = min_proj.min(proj);
            max_proj = max_proj.max(proj);
            min_perp = min_perp.min(perp);
            max_perp = max_perp.max(perp);
        }

        let width = max_proj - min_proj;
        let height = max_perp - min_perp;
        let area = width * height;

        if area < min_area {
            min_area = area;

            let center_proj = (min_proj + max_proj) / 2.0;
            let center_perp = (min_perp + max_perp) / 2.0;

            best = RotatedRect {
                center: Point2f {
                    x: hull[i].x as f32 + center_proj * ux - center_perp * uy,
                    y: hull[i].y as f32 + center_proj * uy + center_perp * ux,
                },
                width,
                height,
                angle: uy.atan2(ux),
            };
        }
    }

    // Ensure width ≥ height (swap if needed and adjust angle)
    if best.width < best.height {
        std::mem::swap(&mut best.width, &mut best.height);
        best.angle += std::f32::consts::FRAC_PI_2;
    }

    best
}

/// Fallback: axis-aligned bounding rectangle.
fn axis_aligned_bounding_rect(points: &[Point2i]) -> RotatedRect {
    let min_x = points.iter().map(|p| p.x).min().unwrap_or(0) as f32;
    let max_x = points.iter().map(|p| p.x).max().unwrap_or(0) as f32;
    let min_y = points.iter().map(|p| p.y).min().unwrap_or(0) as f32;
    let max_y = points.iter().map(|p| p.y).max().unwrap_or(0) as f32;

    RotatedRect {
        center: Point2f {
            x: (min_x + max_x) / 2.0,
            y: (min_y + max_y) / 2.0,
        },
        width: max_x - min_x,
        height: max_y - min_y,
        angle: 0.0,
    }
}

/// Expand (dilate) a rotated rectangle using DBNet's unclip formula.
///
/// Offset distance: d = (Area * ratio) / Perimeter
/// Both width and height are increased by 2 * d.
pub(crate) fn expand_rotated_rect(rect: &RotatedRect, ratio: f32) -> RotatedRect {
    let perimeter = 2.0 * (rect.width + rect.height);
    let d = if perimeter > 1e-4 {
        (rect.width * rect.height * ratio) / perimeter
    } else {
        0.0
    };

    RotatedRect {
        center: rect.center,
        width: rect.width + 2.0 * d,
        height: rect.height + 2.0 * d,
        angle: rect.angle,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convex_hull_square() {
        let points = vec![
            Point2i { x: 0, y: 0 },
            Point2i { x: 10, y: 0 },
            Point2i { x: 10, y: 10 },
            Point2i { x: 0, y: 10 },
            Point2i { x: 5, y: 5 }, // interior point
        ];
        let hull = convex_hull(&points);
        assert_eq!(hull.len(), 4);
    }

    #[test]
    fn test_min_area_rect_axis_aligned() {
        let points = vec![
            Point2i { x: 0, y: 0 },
            Point2i { x: 100, y: 0 },
            Point2i { x: 100, y: 50 },
            Point2i { x: 0, y: 50 },
        ];
        let rect = min_area_rect(&points);
        assert!((rect.width - 100.0).abs() < 2.0 || (rect.height - 100.0).abs() < 2.0);
        assert!(rect.area() > 0.0);
    }

    #[test]
    fn test_rotated_rect_corners() {
        let rect = RotatedRect {
            center: Point2f { x: 50.0, y: 25.0 },
            width: 100.0,
            height: 50.0,
            angle: 0.0,
        };
        let corners = rect.corners();
        // Top-left should be near (0, 0)
        assert!((corners[0][0] - 0.0).abs() < 1.0);
        assert!((corners[0][1] - 0.0).abs() < 1.0);
        // Bottom-right should be near (100, 50)
        assert!((corners[2][0] - 100.0).abs() < 1.0);
        assert!((corners[2][1] - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_simplify_contour() {
        let contour: Vec<Point2i> = (0..100)
            .map(|i| Point2i { x: i, y: 0 })
            .collect();
        let simplified = simplify_contour(&contour, 1.0);
        assert!(simplified.len() < contour.len());
        assert_eq!(simplified.len(), 2); // straight line simplifies to 2 points
    }

    #[test]
    fn test_find_contours_simple() {
        // 5x5 bitmap with a filled 3x3 square in the center
        let mut bitmap = vec![0u8; 25];
        for y in 1..4 {
            for x in 1..4 {
                bitmap[y * 5 + x] = 255;
            }
        }
        let contours = find_contours(&bitmap, 5, 5);
        assert!(!contours.is_empty());
    }
}
