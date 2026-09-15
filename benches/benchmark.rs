//! Criterion benchmarks for SnapOCR pipeline stages.
//!
//! Run with: `cargo bench`
//!
//! Note: Benchmarks that require model files will be skipped if models
//! are not present in the `models/` directory.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

/// Benchmark image preprocessing (resize + normalize).
fn bench_preprocess(c: &mut Criterion) {
    // Create a synthetic 1920x1080 RGB image
    let width = 1920u32;
    let height = 1080u32;
    let img = image::RgbImage::from_fn(width, height, |x, y| {
        image::Rgb([
            ((x * 7 + y * 13) % 256) as u8,
            ((x * 11 + y * 3) % 256) as u8,
            ((x * 5 + y * 17) % 256) as u8,
        ])
    });
    let dynamic_img = image::DynamicImage::ImageRgb8(img);

    c.bench_function("preprocess_1080p", |b| {
        b.iter(|| {
            // We can't call the internal preprocess function directly,
            // so we benchmark the image operations that constitute it.
            let rgb = dynamic_img.to_rgb8();
            let _resized = image::imageops::resize(&rgb, 960, 544, image::imageops::FilterType::Triangle);
            black_box(());
        });
    });
}

/// Benchmark RGBA to grayscale conversion.
fn bench_rgba_decode(c: &mut Criterion) {
    let width = 1920u32;
    let height = 1080u32;
    let rgba = vec![128u8; (width * height * 4) as usize];

    c.bench_function("rgba_decode_1080p", |b| {
        b.iter(|| {
            let img = image::RgbaImage::from_raw(width, height, rgba.clone()).unwrap();
            let _rgb = image::DynamicImage::ImageRgba8(img).to_rgb8();
            black_box(());
        });
    });
}

/// Benchmark image resize operation.
fn bench_resize(c: &mut Criterion) {
    let img = image::RgbImage::from_fn(1920, 1080, |x, y| {
        image::Rgb([((x + y) % 256) as u8; 3])
    });

    c.bench_function("resize_1080p_to_960", |b| {
        b.iter(|| {
            let _resized =
                image::imageops::resize(&img, 960, 544, image::imageops::FilterType::Triangle);
            black_box(());
        });
    });
}

/// Benchmark CTC greedy decoding.
fn bench_ctc_decode(c: &mut Criterion) {
    // Simulate a typical CRNN output: seq_len=80, vocab_size=6700
    let seq_len = 80;
    let vocab_size = 6700;
    let logits: Vec<f32> = (0..seq_len * vocab_size)
        .map(|i| ((i % 100) as f32 - 50.0) / 10.0)
        .collect();

    c.bench_function("ctc_decode_80x6700", |b| {
        b.iter(|| {
            // Inline CTC decode logic for benchmarking
            let mut char_indices = Vec::with_capacity(seq_len);
            let mut prev = 0usize;
            for t in 0..seq_len {
                let row = &logits[t * vocab_size..(t + 1) * vocab_size];
                let mut max_idx = 0;
                let mut max_val = f32::NEG_INFINITY;
                for (i, &v) in row.iter().enumerate() {
                    if v > max_val {
                        max_val = v;
                        max_idx = i;
                    }
                }
                if max_idx != 0 && max_idx != prev {
                    char_indices.push(max_idx);
                }
                prev = max_idx;
            }
            black_box(char_indices);
        });
    });
}

/// Benchmark text cleanup.
fn bench_text_cleanup(c: &mut Criterion) {
    let input = "你 好 ，world！这是 一个 test 测试 123 。 Hello World ！";

    c.bench_function("text_cleanup_mixed", |b| {
        b.iter(|| {
            // Simulate text cleanup operations
            let mut result = String::with_capacity(input.len());
            let chars: Vec<char> = input.chars().collect();
            for &ch in &chars {
                let normalized = match ch {
                    '\u{FF01}'..='\u{FF5E}' => {
                        (ch as u32 - 0xFF01 + 0x21) as u8 as char
                    }
                    '\u{3000}' => ' ',
                    _ => ch,
                };
                result.push(normalized);
            }
            black_box(result);
        });
    });
}

criterion_group!(
    benches,
    bench_preprocess,
    bench_rgba_decode,
    bench_resize,
    bench_ctc_decode,
    bench_text_cleanup,
);
criterion_main!(benches);
