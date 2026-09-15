# SnapOCR — 开源轻量快速桌面 OCR 引擎

> **Sub-100ms** 纯 CPU 中英混排文字识别，专为桌面截图场景优化

## 特性

- 🦀 **纯 Rust 实现** — 使用 `tract-onnx`，零 C/C++ 外部依赖
- ⚡ **极速推理** — 常规段落 (5~15 行) 识别 ≤ 120ms，单行 ≤ 60ms
- 🌐 **中英混排** — 统一字符集，杜绝 `I → 丨` / `writi ng` 类错误
- 📦 **轻量级** — 常驻内存 ≤ 40MB，模型体积 ≤ 25MB
- 🖥️ **跨平台** — Windows / macOS / Linux (x86_64, ARM64)
- 🔌 **双接口** — Rust lib (Tauri 直接依赖) + C-ABI FFI (动态链接库)

## 快速开始

### 作为 Tauri 依赖引入

```toml
# 在你的 Tauri 项目的 Cargo.toml 中添加
[dependencies]
snap-ocr = { path = "../snap-ocr" }
```

### Rust API

```rust
use snap_ocr::{SnapOcrEngine, OcrEngineConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = OcrEngineConfig {
        model_dir: "models/".into(),
        num_threads: 4,
        ..Default::default()
    };

    let engine = SnapOcrEngine::new(config)?;

    // 从 RGBA 像素识别（截图场景）
    let rgba_data: Vec<u8> = capture_screen(); // 你的截图函数
    let result = engine.recognize_rgba(&rgba_data, 1920, 1080)?;

    println!("识别文本:\n{}", result.text);
    println!("耗时: {:.1}ms", result.latency_ms.total_ms);

    for line in &result.lines {
        println!("[{:.2}] {}", line.score, line.text);
    }

    Ok(())
}
```

### C-ABI FFI

```c
#include "snap_ocr.h"

snap_ocr_handle_t handle = snap_ocr_init("models/", 4);
snap_ocr_response_t* resp = snap_ocr_infer_rgba(handle, rgba_data, width, height);

printf("Text: %s\n", resp->full_text);
printf("Time: %.1f ms\n", resp->total_ms);

snap_ocr_free_response(resp);
snap_ocr_destroy(handle);
```

## 模型文件

将以下模型文件放入 `models/` 目录：

| 文件 | 说明 | 格式 |
|:---|:---|:---|
| `det.onnx` | DBNet 文本检测模型 | ONNX (FP16/INT8, ≤ 3MB) |
| `rec.onnx` | CRNN 文本识别模型 | ONNX (INT8, ≤ 12MB) |
| `vocab.txt` | 中英统一字符字典 | 文本文件，每行一个字符 |

### 兼容模型

可使用 PaddleOCR 预训练模型转换后直接使用：

```bash
# 下载 PaddleOCR 轻量模型
# det: ch_PP-OCRv4_det (转换为 ONNX 后重命名为 det.onnx)
# rec: ch_PP-OCRv4_rec (转换为 ONNX 后重命名为 rec.onnx)
# vocab: ppocr_keys_v1.txt (重命名为 vocab.txt)
```

## 架构

```
RGBA/PNG 输入
  → 前处理 (灰度/缩放/ImageNet归一化/NCHW张量)
  → DBNet 文本检测 (概率图→二值化→轮廓提取→旋转矩形)
  → 透视变换裁剪 (文本行矫正为水平条带)
  → CRNN 批处理识别 (CNN+BiLSTM+CTC解码)
  → 后处理 (段落几何重排 + 中英混排清洗)
  → OcrResult {text, lines[], latency}
```

## 构建

```bash
# 编译 Release 版本
cargo build --release

# 运行单元测试
cargo test --lib

# 运行集成测试（需要模型文件）
cargo test --test integration_test -- --ignored

# 基准测试
cargo bench

# 代码检查
cargo clippy -- -D warnings
```

## 目录结构

```
snap-ocr/
├── Cargo.toml
├── src/
│   ├── lib.rs              # 公开 API 入口
│   ├── engine.rs           # SnapOcrEngine 核心
│   ├── config.rs           # 配置
│   ├── types.rs            # 类型定义
│   ├── error.rs            # 错误类型
│   ├── preprocess.rs       # 图像前处理
│   ├── crop.rs             # 仿射变换裁剪
│   ├── ffi.rs              # C-ABI FFI
│   ├── detect/
│   │   ├── dbnet.rs        # DBNet 推理 + 后处理
│   │   └── contour.rs      # 纯 Rust 轮廓提取
│   ├── recognize/
│   │   ├── crnn.rs         # CRNN 推理
│   │   └── ctc_decode.rs   # CTC 贪心解码
│   └── postprocess/
│       ├── paragraph.rs    # 段落几何重排
│       └── text_clean.rs   # 中英混排清洗
├── models/                 # ONNX 模型目录
├── tests/                  # 集成测试
└── benches/                # 基准测试
```

## 许可证

MIT OR Apache-2.0
