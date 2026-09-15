//! Paragraph geometric reordering algorithm.
//!
//! Reconstructs natural reading order from arbitrarily-ordered detected
//! text line bounding boxes. Handles multi-column layouts, paragraph
//! detection, and proper line sequencing.

use crate::types::OcrLineResult;

/// Reorder detected text lines into natural reading order and group into paragraphs.
///
/// # Algorithm
/// 1. **Y-coordinate clustering**: Group lines whose vertical centers are within
///    half a line height of each other into logical rows.
/// 2. **X-coordinate sorting**: Within each logical row, sort lines left-to-right.
/// 3. **Row sorting**: Sort logical rows top-to-bottom.
/// 4. **Paragraph detection**: Split into paragraphs when the vertical gap between
///    adjacent rows exceeds 1.5× the average line height.
///
/// # Returns
/// The input lines reordered into reading order, plus the combined text
/// with paragraph breaks (double newline) and line breaks (single newline).
pub(crate) fn reorder_into_paragraphs(lines: &mut Vec<OcrLineResult>) -> String {
    if lines.is_empty() {
        return String::new();
    }

    if lines.len() == 1 {
        return lines[0].text.clone();
    }

    // Step 1: Compute line metrics
    let metrics: Vec<LineMetrics> = lines
        .iter()
        .enumerate()
        .map(|(idx, line)| {
            let center = line.box_coords.center();
            let height = line.box_coords.height();
            LineMetrics {
                index: idx,
                center_x: center[0],
                center_y: center[1],
                left_x: min_x(&line.box_coords.points),
                height,
            }
        })
        .collect();

    // Step 2: Cluster lines into logical rows by Y-coordinate proximity
    let avg_height = metrics.iter().map(|m| m.height).sum::<f32>() / metrics.len() as f32;
    let row_threshold = avg_height * 0.5;

    let mut sorted_by_y: Vec<usize> = (0..metrics.len()).collect();
    sorted_by_y.sort_by(|&a, &b| {
        metrics[a]
            .center_y
            .partial_cmp(&metrics[b].center_y)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut rows: Vec<Vec<usize>> = Vec::new();
    let mut current_row: Vec<usize> = vec![sorted_by_y[0]];
    let mut current_row_y = metrics[sorted_by_y[0]].center_y;

    for &idx in sorted_by_y.iter().skip(1) {
        if (metrics[idx].center_y - current_row_y).abs() <= row_threshold {
            // Same logical row
            current_row.push(idx);
        } else {
            // New row
            rows.push(current_row);
            current_row = vec![idx];
            current_row_y = metrics[idx].center_y;
        }
    }
    rows.push(current_row);

    // Step 3: Sort lines within each row by X-coordinate (left to right)
    for row in &mut rows {
        row.sort_by(|&a, &b| {
            metrics[a]
                .center_x
                .partial_cmp(&metrics[b].center_x)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    // Step 4: Sort rows by the average Y-coordinate of their lines
    rows.sort_by(|a, b| {
        let avg_y_a: f32 = a.iter().map(|&i| metrics[i].center_y).sum::<f32>() / a.len() as f32;
        let avg_y_b: f32 = b.iter().map(|&i| metrics[i].center_y).sum::<f32>() / b.len() as f32;
        avg_y_a
            .partial_cmp(&avg_y_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Step 5: Detect paragraph breaks and build output
    let paragraph_gap_threshold = avg_height * 1.5;

    let mut paragraphs: Vec<Vec<String>> = Vec::new();
    let mut current_paragraph: Vec<String> = Vec::new();
    let mut prev_row_y: Option<f32> = None;

    // Also reorder the lines vector to match reading order
    let mut reordered_lines: Vec<OcrLineResult> = Vec::with_capacity(lines.len());

    for row in &rows {
        // Calculate average Y for this row
        let row_avg_y =
            row.iter().map(|&i| metrics[i].center_y).sum::<f32>() / row.len() as f32;

        // Check for paragraph break
        if let Some(prev_y) = prev_row_y {
            if (row_avg_y - prev_y) > paragraph_gap_threshold {
                // Paragraph break detected
                if !current_paragraph.is_empty() {
                    paragraphs.push(current_paragraph);
                    current_paragraph = Vec::new();
                }
            }
        }

        // Concatenate text within the same logical row
        let row_text: String = row
            .iter()
            .map(|&i| lines[i].text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        current_paragraph.push(row_text);

        // Collect reordered lines
        for &i in row {
            reordered_lines.push(lines[i].clone());
        }

        prev_row_y = Some(row_avg_y);
    }

    if !current_paragraph.is_empty() {
        paragraphs.push(current_paragraph);
    }

    // Update the lines vector with reading order
    *lines = reordered_lines;

    // Build final text: lines within a paragraph joined by \n, paragraphs by \n\n
    paragraphs
        .iter()
        .map(|p| p.join("\n"))
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Internal metrics for a detected text line.
#[derive(Debug)]
#[allow(dead_code)]
struct LineMetrics {
    /// Original index in the lines vector.
    index: usize,
    /// X-coordinate of the bounding box center.
    center_x: f32,
    /// Y-coordinate of the bounding box center.
    center_y: f32,
    /// Leftmost X-coordinate of the bounding box.
    left_x: f32,
    /// Height of the bounding box.
    height: f32,
}

/// Find the minimum X value among 4 corner points.
#[inline]
fn min_x(points: &[[f32; 2]; 4]) -> f32 {
    points
        .iter()
        .map(|p| p[0])
        .fold(f32::MAX, f32::min)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{OcrBoundingBox, OcrLineResult};

    fn make_line(text: &str, x: f32, y: f32, w: f32, h: f32) -> OcrLineResult {
        OcrLineResult {
            text: text.to_string(),
            box_coords: OcrBoundingBox::from_rect(x, y, w, h),
            score: 0.95,
        }
    }

    #[test]
    fn test_single_line() {
        let mut lines = vec![make_line("hello world", 10.0, 10.0, 200.0, 20.0)];
        let text = reorder_into_paragraphs(&mut lines);
        assert_eq!(text, "hello world");
    }

    #[test]
    fn test_two_lines_ordered() {
        let mut lines = vec![
            make_line("first line", 10.0, 10.0, 200.0, 20.0),
            make_line("second line", 10.0, 35.0, 200.0, 20.0),
        ];
        let text = reorder_into_paragraphs(&mut lines);
        assert_eq!(text, "first line\nsecond line");
    }

    #[test]
    fn test_two_lines_reversed() {
        let mut lines = vec![
            make_line("second line", 10.0, 35.0, 200.0, 20.0),
            make_line("first line", 10.0, 10.0, 200.0, 20.0),
        ];
        let text = reorder_into_paragraphs(&mut lines);
        assert_eq!(text, "first line\nsecond line");
    }

    #[test]
    fn test_paragraph_break() {
        let mut lines = vec![
            make_line("paragraph 1 line 1", 10.0, 10.0, 300.0, 20.0),
            make_line("paragraph 1 line 2", 10.0, 35.0, 300.0, 20.0),
            // Gap > 1.5 * avg_height (~30px) → paragraph break
            make_line("paragraph 2 line 1", 10.0, 100.0, 300.0, 20.0),
        ];
        let text = reorder_into_paragraphs(&mut lines);
        assert!(text.contains("\n\n"), "Should have paragraph break");
        assert!(text.starts_with("paragraph 1 line 1"));
    }

    #[test]
    fn test_horizontal_ordering() {
        // Two lines on the same row, should be ordered left-to-right
        let mut lines = vec![
            make_line("right", 200.0, 10.0, 100.0, 20.0),
            make_line("left", 10.0, 10.0, 100.0, 20.0),
        ];
        let text = reorder_into_paragraphs(&mut lines);
        assert_eq!(text, "left right");
    }

    #[test]
    fn test_multiline_with_inline() {
        // 3 rows, with the middle row having 2 inline items
        let mut lines = vec![
            make_line("top", 10.0, 10.0, 200.0, 20.0),
            make_line("mid-left", 10.0, 35.0, 100.0, 20.0),
            make_line("mid-right", 130.0, 35.0, 100.0, 20.0),
            make_line("bottom", 10.0, 60.0, 200.0, 20.0),
        ];
        let text = reorder_into_paragraphs(&mut lines);
        let text_lines: Vec<&str> = text.lines().collect();
        assert_eq!(text_lines[0], "top");
        assert_eq!(text_lines[1], "mid-left mid-right");
        assert_eq!(text_lines[2], "bottom");
    }
}
