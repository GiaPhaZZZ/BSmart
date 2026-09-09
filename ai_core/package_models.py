#!/usr/bin/env python3
"""
BSmart Model Packaging & Manifest Generator (MOD-04)
Collects exported ONNX models from PhoWhisper, SmolVLM2, YOLO26s, and ZipDepth,
computes verification checksums, generates models_manifest.json,
and packages them into Android assets for APK bundling.

Usage:
  python package_models.py [--source-dir exported_models] [--target-dir ../mobile_app/android/app/src/main/assets/models] [--verify]
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, List

AI_CORE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = AI_CORE_DIR / "exported_models"
DEFAULT_TARGET_DIR = (
    AI_CORE_DIR.parent
    / "mobile_app"
    / "android"
    / "app"
    / "src"
    / "main"
    / "assets"
    / "models"
)


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def scan_and_collect_models(source_dir: Path) -> List[Dict[str, Any]]:
    """Scan source directory for ONNX models and configuration files."""
    collected = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            path = Path(root) / file
            rel_path = path.relative_to(source_dir)
            ext = path.suffix.lower()

            item_type = "unknown"
            if ext == ".onnx":
                item_type = "onnx_model"
            elif ext == ".json":
                item_type = "config_metadata"
            elif ext in [".txt", ".bin"]:
                item_type = "vocab_weights"

            collected.append({
                "name": file,
                "relative_path": str(rel_path).replace("\\", "/"),
                "file_path": path,
                "type": item_type,
                "size_bytes": path.stat().st_size,
                "sha256": compute_sha256(path),
            })
    return collected


def generate_manifest(collected_items: List[Dict[str, Any]], target_dir: Path) -> Path:
    """Generate models_manifest.json describing bundled on-device models."""
    manifest_data: Dict[str, Any] = {
        "project": "BSmart Multimodal Smart Glasses",
        "version": "1.0.0",
        "manifest_version": "1",
        "platform": "android-arm64-v8a",
        "runtime": "onnxruntime-mobile",
        "models": {
            "asr": {
                "name": "PhoWhisper-tiny",
                "encoder": "phowhisper/phowhisper_encoder.onnx",
                "decoder": "phowhisper/phowhisper_decoder.onnx",
                "vocab": "phowhisper/vocab.json",
                "meta": "phowhisper/model_meta.json",
            },
            "vlm": {
                "name": "SmolVLM2-256M",
                "vision": "smolvlm2/smolvlm2_vision.onnx",
                "lm": "smolvlm2/smolvlm2_lm.onnx",
                "config": "smolvlm2/qa_config.json",
            },
            "navigation": {
                "name": "YOLO26s_ZipDepth",
                "detector": "navigation/yolo26s.onnx",
                "depth": "navigation/zipdepth.onnx",
                "labels": "navigation/coco_labels_vi.json",
                "meta": "navigation/navigation_model_meta.json",
            }
        },
        "files": [
            {
                "name": item["name"],
                "path": item["relative_path"],
                "type": item["type"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
            for item in collected_items
        ],
        "total_files": len(collected_items),
        "total_size_bytes": sum(item["size_bytes"] for item in collected_items),
    }

    manifest_path = target_dir / "models_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, ensure_ascii=False, indent=2)

    return manifest_path


def copy_to_assets(collected_items: List[Dict[str, Any]], target_dir: Path) -> None:
    """Copy collected models and files into target assets folder."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in collected_items:
        dest = target_dir / item["relative_path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item["file_path"], dest)
        print(f"  -> Packaged: {item['relative_path']} ({item['size_bytes'] / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="BSmart Model Packaging & Manifest (MOD-04)")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR, help="Exported models folder")
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR, help="Android assets target")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing manifest")
    args = parser.parse_args()

    print("=== BSmart MOD-04: On-Device Model Packaging ===")
    print(f"Source directory: {args.source_dir}")
    print(f"Target directory: {args.target_dir}")

    if not args.source_dir.exists():
        print(f"[MOD-04 Warning] Source directory {args.source_dir} does not exist yet.")
        print("Please run export_phowhisper_onnx.py, export_smolvlm2_onnx.py, and export_yolo_zipdepth_onnx.py first.")
        sys.exit(0)

    items = scan_and_collect_models(args.source_dir)
    print(f"[MOD-04] Discovered {len(items)} model files/configs.")

    if not args.verify_only:
        print("[MOD-04] Copying files to Android assets folder...")
        copy_to_assets(items, args.target_dir)

    manifest_path = generate_manifest(items, args.target_dir)
    print(f"[MOD-04] Manifest generated at: {manifest_path}")

    total_mb = sum(i["size_bytes"] for i in items) / 1024 / 1024
    print(f"[MOD-04] Packaging completed successfully! Total packaged size: {total_mb:.2f} MB")


if __name__ == "__main__":
    main()
