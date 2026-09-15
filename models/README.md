# SnapOCR Models

Place the following ONNX models in this directory:

- `det.onnx`: DBNet text detection model (~4.7MB, ONNX format, dynamic shape `[1, 3, H, W]`).
- `rec.onnx`: CRNN text recognition model (~10.8MB, ONNX format, dynamic width `[1, 3, 48, W]`).
- `vocab.txt`: Character vocabulary (6,624 characters + CTC blank and space tokens).

These models are derived from PaddleOCR v4 lightweight models (`ch_PP-OCRv4_det` and `ch_PP-OCRv4_rec`) converted via `paddle2onnx`.
