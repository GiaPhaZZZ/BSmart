#!/usr/bin/env python3
"""
BSmart AI Model Export — YOLO26s + ZipDepth to ONNX (MOD-03)
Converts Object Detection (YOLO26s) and Monocular Depth Estimation (ZipDepth)
into ONNX format for on-device Feature 3: Navigation / Obstacle Awareness.

Outputs:
  - yolo26s.onnx (Object Detection, input: [1, 3, 640, 640])
  - zipdepth.onnx (Depth Estimation, input: [1, 3, 384, 384])
  - coco_labels_vi.json (80 COCO classes with Vietnamese translations)
  - navigation_model_meta.json (Preprocessing parameters & detection anchors)

Usage:
  python export_yolo_zipdepth_onnx.py [--output-dir ../models/navigation] [--verify]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "exported_models" / "navigation"
DEFAULT_YOLO_WEIGHTS = PROJECT_ROOT / "models" / "yolo26s.pt"
DEFAULT_ZIPDEPTH_WEIGHTS = PROJECT_ROOT / "models" / "ZipDepth" / "checkpoints" / "zipdepth_base_npu.pth"
ZIPDEPTH_EXPORT_SCRIPT = PROJECT_ROOT / "models" / "ZipDepth" / "scripts" / "export.py"

COCO_CLASSES_VI: List[Dict[str, Any]] = [
    {"id": 0, "en": "person", "vi": "người", "priority": 1},
    {"id": 1, "en": "bicycle", "vi": "xe đạp", "priority": 2},
    {"id": 2, "en": "car", "vi": "ô tô", "priority": 1},
    {"id": 3, "en": "motorcycle", "vi": "xe máy", "priority": 1},
    {"id": 4, "en": "airplane", "vi": "máy bay", "priority": 3},
    {"id": 5, "en": "bus", "vi": "xe buýt", "priority": 1},
    {"id": 6, "en": "train", "vi": "tàu hỏa", "priority": 1},
    {"id": 7, "en": "truck", "vi": "xe tải", "priority": 1},
    {"id": 8, "en": "boat", "vi": "thuyền", "priority": 3},
    {"id": 9, "en": "traffic light", "vi": "đèn giao thông", "priority": 2},
    {"id": 10, "en": "fire hydrant", "vi": "trụ cứu hỏa", "priority": 2},
    {"id": 11, "en": "stop sign", "vi": "biển báo dừng", "priority": 2},
    {"id": 12, "en": "parking meter", "vi": "đồng hồ đỗ xe", "priority": 3},
    {"id": 13, "en": "bench", "vi": "ghế dài", "priority": 2},
    {"id": 14, "en": "bird", "vi": "con chim", "priority": 3},
    {"id": 15, "en": "cat", "vi": "con mèo", "priority": 3},
    {"id": 16, "en": "dog", "vi": "con chó", "priority": 2},
    {"id": 17, "en": "horse", "vi": "con ngựa", "priority": 3},
    {"id": 18, "en": "sheep", "vi": "con cừu", "priority": 3},
    {"id": 19, "en": "cow", "vi": "con bò", "priority": 2},
    {"id": 20, "en": "elephant", "vi": "con voi", "priority": 3},
    {"id": 21, "en": "bear", "vi": "con gấu", "priority": 3},
    {"id": 22, "en": "zebra", "vi": "ngựa vằn", "priority": 3},
    {"id": 23, "en": "giraffe", "vi": "hươu cao cổ", "priority": 3},
    {"id": 24, "en": "backpack", "vi": "ba lô", "priority": 3},
    {"id": 25, "en": "umbrella", "vi": "chiếc ô", "priority": 3},
    {"id": 26, "en": "handbag", "vi": "túi xách", "priority": 3},
    {"id": 27, "en": "tie", "vi": "cà vạt", "priority": 3},
    {"id": 28, "en": "suitcase", "vi": "vali", "priority": 3},
    {"id": 29, "en": "frisbee", "vi": "đĩa bay", "priority": 3},
    {"id": 30, "en": "skis", "vi": "ván trượt tuyết", "priority": 3},
    {"id": 31, "en": "snowboard", "vi": "ván trượt", "priority": 3},
    {"id": 32, "en": "sports ball", "vi": "quả bóng", "priority": 3},
    {"id": 33, "en": "kite", "vi": "con diều", "priority": 3},
    {"id": 34, "en": "baseball bat", "vi": "gậy bóng chày", "priority": 3},
    {"id": 35, "en": "baseball glove", "vi": "găng tay bóng chày", "priority": 3},
    {"id": 36, "en": "skateboard", "vi": "ván trượt", "priority": 3},
    {"id": 37, "en": "surfboard", "vi": "ván lướt sóng", "priority": 3},
    {"id": 38, "en": "tennis racket", "vi": "vợt tennis", "priority": 3},
    {"id": 39, "en": "bottle", "vi": "chai nước", "priority": 3},
    {"id": 40, "en": "wine glass", "vi": "ly thủy tinh", "priority": 3},
    {"id": 41, "en": "cup", "vi": "chiếc cốc", "priority": 3},
    {"id": 42, "en": "fork", "vi": "dĩa", "priority": 3},
    {"id": 43, "en": "knife", "vi": "con dao", "priority": 3},
    {"id": 44, "en": "spoon", "vi": "chiếc thìa", "priority": 3},
    {"id": 45, "en": "bowl", "vi": "chiếc bát", "priority": 3},
    {"id": 46, "en": "banana", "vi": "quả chuối", "priority": 3},
    {"id": 47, "en": "apple", "vi": "quả táo", "priority": 3},
    {"id": 48, "en": "sandwich", "vi": "bánh mì kẹp", "priority": 3},
    {"id": 49, "en": "orange", "vi": "quả cam", "priority": 3},
    {"id": 50, "en": "broccoli", "vi": "bông cải xanh", "priority": 3},
    {"id": 51, "en": "carrot", "vi": "củ cà rốt", "priority": 3},
    {"id": 52, "en": "hot dog", "vi": "bánh xúc xích", "priority": 3},
    {"id": 53, "en": "pizza", "vi": "bánh pizza", "priority": 3},
    {"id": 54, "en": "donut", "vi": "bánh donut", "priority": 3},
    {"id": 55, "en": "cake", "vi": "bánh ngọt", "priority": 3},
    {"id": 56, "en": "chair", "vi": "cái ghế", "priority": 2},
    {"id": 57, "en": "couch", "vi": "ghế sofa", "priority": 2},
    {"id": 58, "en": "potted plant", "vi": "chậu cây", "priority": 2},
    {"id": 59, "en": "bed", "vi": "chiếc giường", "priority": 2},
    {"id": 60, "en": "dining table", "vi": "bàn ăn", "priority": 2},
    {"id": 61, "en": "toilet", "vi": "bồn cầu", "priority": 2},
    {"id": 62, "en": "tv", "vi": "tivi", "priority": 3},
    {"id": 63, "en": "laptop", "vi": "máy tính xách tay", "priority": 3},
    {"id": 64, "en": "mouse", "vi": "chuột máy tính", "priority": 3},
    {"id": 65, "en": "remote", "vi": "điều khiển từ xa", "priority": 3},
    {"id": 66, "en": "keyboard", "vi": "bàn phím", "priority": 3},
    {"id": 67, "en": "cell phone", "vi": "điện thoại", "priority": 3},
    {"id": 68, "en": "microwave", "vi": "lò vi sóng", "priority": 3},
    {"id": 69, "en": "oven", "vi": "lò nướng", "priority": 3},
    {"id": 70, "en": "toaster", "vi": "máy nướng bánh", "priority": 3},
    {"id": 71, "en": "sink", "vi": "bồn rửa", "priority": 2},
    {"id": 72, "en": "refrigerator", "vi": "tủ lạnh", "priority": 2},
    {"id": 73, "en": "book", "vi": "cuốn sách", "priority": 3},
    {"id": 74, "en": "clock", "vi": "đồng hồ", "priority": 3},
    {"id": 75, "en": "vase", "vi": "lọ hoa", "priority": 3},
    {"id": 76, "en": "scissors", "vi": "cây kéo", "priority": 3},
    {"id": 77, "en": "teddy bear", "vi": "gấu bông", "priority": 3},
    {"id": 78, "en": "hair drier", "vi": "máy sấy tóc", "priority": 3},
    {"id": 79, "en": "toothbrush", "vi": "bàn chải đánh răng", "priority": 3},
]


def export_metadata(output_dir: Path) -> Path:
    """Export Vietnamese labels and navigation model hyperparameters."""
    labels_path = output_dir / "coco_labels_vi.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(COCO_CLASSES_VI, f, ensure_ascii=False, indent=2)

    meta = {
        "models": {
            "yolo": {
                "name": "yolo26s",
                "filename": "yolo26s.onnx",
                "input_name": "images",
                "output_name": "output0",
                "input_shape": [1, 3, 640, 640],
                "output_shape": [1, 300, 6],
                "output_format": "xyxy_conf_class",
                "opset": 14,
                "confidence_threshold": 0.5,
                "iou_threshold": 0.45,
                "num_classes": 80,
            },
            "zipdepth": {
                "name": "zipdepth_base_npu",
                "filename": "zipdepth.onnx",
                "input_name": "image",
                "output_name": "depth",
                "input_shape": [1, 3, 384, 384],
                "output_shape": [1, 1, 384, 384],
                "opset": 18,
                "external_data": False,
                "scale_type": "relative_inverse_depth",
                "near_threshold": 0.4,
            }
        },
        "navigation_rules": {
            "grid_horizontal": {"left": 0.33, "center": 0.66, "right": 1.0},
        "cooldown_seconds": 8,
        "cycle_seconds": 4,
        "max_reported_objects": 2,
        "template": "Chú ý, có {class} ở {position}, {distance}"
    }
    }

    meta_path = output_dir / "navigation_model_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta_path


def export_yolo(output_dir: Path, weights_path: Path = DEFAULT_YOLO_WEIGHTS) -> Path:
    """Export YOLO26s model to ONNX."""
    output_path = output_dir / "yolo26s.onnx"
    print(f"[MOD-03] Exporting YOLO26s to {output_path}...")

    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

    from ultralytics import YOLO

    print(f"[MOD-03] Loading real YOLO weights from {weights_path}...")
    model = YOLO(str(weights_path))
    exported = Path(model.export(format="onnx", imgsz=640, opset=14, dynamic=False, simplify=False, nms=False))
    if not exported.exists():
        exported = weights_path.with_suffix(".onnx")
    if not exported.exists():
        raise RuntimeError(f"Ultralytics export did not create an ONNX file for {weights_path}")

    os.replace(exported, output_path)
    print(f"[MOD-03] Ultralytics export complete: {output_path} ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
    return output_path


def export_zipdepth(output_dir: Path, weights_path: Path = DEFAULT_ZIPDEPTH_WEIGHTS) -> Path:
    """Export ZipDepth model to ONNX."""
    output_path = output_dir / "zipdepth.onnx"
    output_path_abs = output_path.resolve()
    temp_output_path = output_path_abs.with_name("zipdepth_export_tmp.onnx")
    print(f"[MOD-03] Exporting ZipDepth to {output_path}...")

    if not weights_path.exists():
        raise FileNotFoundError(f"ZipDepth checkpoint not found: {weights_path}")
    if not ZIPDEPTH_EXPORT_SCRIPT.exists():
        raise FileNotFoundError(f"ZipDepth exporter not found: {ZIPDEPTH_EXPORT_SCRIPT}")

    cmd = [
        sys.executable,
        str(ZIPDEPTH_EXPORT_SCRIPT),
        "--ckpt",
        str(weights_path),
        "--format",
        "onnx",
        "--variant",
        "base",
        "--global-mode",
        "balanced",
        "--height",
        "384",
        "--width",
        "384",
        "--opset",
        "18",
        "--npu",
        "--output",
        str(temp_output_path),
    ]
    for stale in [
        output_path_abs,
        temp_output_path,
        temp_output_path.with_name(f"{temp_output_path.stem}_raw.onnx"),
        temp_output_path.with_name(f"{temp_output_path.stem}_raw.onnx.data"),
    ]:
        if stale.exists():
            stale.unlink()

    print(f"[MOD-03] Running real ZipDepth exporter: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    subprocess.run(cmd, cwd=str(ZIPDEPTH_EXPORT_SCRIPT.parent.parent), env=env, check=True)
    if not temp_output_path.exists():
        raise RuntimeError(f"ZipDepth export did not create {temp_output_path}")

    import onnx

    model = onnx.load(str(temp_output_path), load_external_data=True)
    onnx.save_model(model, str(output_path_abs), save_as_external_data=False)
    for stale in [
        temp_output_path,
        temp_output_path.with_name(f"{temp_output_path.stem}.data"),
        temp_output_path.with_name(f"{temp_output_path.stem}_raw.onnx"),
        temp_output_path.with_name(f"{temp_output_path.stem}_raw.onnx.data"),
    ]:
        if stale.exists():
            stale.unlink()

    if not output_path.exists():
        raise RuntimeError(f"ZipDepth export did not create {output_path}")

    print(f"[MOD-03] ZipDepth exported successfully ({output_path.stat().st_size / 1024 / 1024:.2f} MB)")
    return output_path


def verify_export(yolo_path: Path, depth_path: Path) -> bool:
    """Verify exported ONNX models."""
    print("[MOD-03] Verifying YOLO26s and ZipDepth ONNX models...")
    try:
        import onnx
        onnx.checker.check_model(str(yolo_path))
        onnx.checker.check_model(str(depth_path))
        print("[MOD-03] ONNX checker passed for both YOLO and ZipDepth!")
        return True
    except ImportError:
        print("[MOD-03] onnx package not installed, file existence and size verified:")
        print(f"  - {yolo_path.name}: {yolo_path.stat().st_size} bytes")
        print(f"  - {depth_path.name}: {depth_path.stat().st_size} bytes")
        return yolo_path.exists() and depth_path.exists()


def main():
    parser = argparse.ArgumentParser(description="Export YOLO26s + ZipDepth to ONNX (MOD-03)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Destination directory")
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO_WEIGHTS, help="yolo26s.pt path")
    parser.add_argument("--zipdepth-weights", type=Path, default=DEFAULT_ZIPDEPTH_WEIGHTS, help="zipdepth.pth path")
    parser.add_argument("--verify", action="store_true", default=True, help="Verify exported ONNX graph")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== BSmart MOD-03: YOLO26s + ZipDepth ONNX Export ===")
    print(f"Target directory: {args.output_dir}")

    meta_path = export_metadata(args.output_dir)
    yolo_path = export_yolo(args.output_dir, args.yolo_weights)
    depth_path = export_zipdepth(args.output_dir, args.zipdepth_weights)

    if args.verify:
        verify_export(yolo_path, depth_path)

    print(f"[MOD-03] Completed! Models and metadata ready in: {args.output_dir}")


if __name__ == "__main__":
    main()
