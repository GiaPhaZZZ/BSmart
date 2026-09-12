import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def resize_keeping_aspect(image: Image.Image, longest_edge: int) -> Image.Image:
    width, height = image.size
    scale = longest_edge / float(max(width, height))
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    return image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)


def resize_for_vision_encoder(image: Image.Image, max_image_size: int) -> Image.Image:
    width, height = image.size
    aspect_ratio = width / float(height)
    if width >= height:
        resized_width = int(np.ceil(width / max_image_size) * max_image_size)
        aspect_height = int(resized_width / aspect_ratio)
        resized_height = int(np.ceil(aspect_height / max_image_size) * max_image_size)
    else:
        resized_height = int(np.ceil(height / max_image_size) * max_image_size)
        aspect_width = int(resized_height * aspect_ratio)
        resized_width = int(np.ceil(aspect_width / max_image_size) * max_image_size)
    return image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)


def build_image_inputs(image_path: Path):
    max_image_size = 512
    image = Image.open(image_path).convert("RGB")
    prompt_sized = resize_keeping_aspect(image, 2048)
    vision_sized = resize_for_vision_encoder(prompt_sized, max_image_size)

    width, height = vision_sized.size
    frames = []
    rows = 0
    cols = 0
    if height > max_image_size or width > max_image_size:
        rows = int(np.ceil(height / max_image_size))
        cols = int(np.ceil(width / max_image_size))
        optimal_height = int(np.ceil(height / rows))
        optimal_width = int(np.ceil(width / cols))
        for row in range(rows):
            for col in range(cols):
                start_x = col * optimal_width
                start_y = row * optimal_height
                end_x = min(start_x + optimal_width, width)
                end_y = min(start_y + optimal_height, height)
                crop = vision_sized.crop((start_x, start_y, end_x, end_y))
                if crop.size != (max_image_size, max_image_size):
                    crop = crop.resize((max_image_size, max_image_size), Image.Resampling.BICUBIC)
                frames.append(crop)
        frames.append(vision_sized.resize((max_image_size, max_image_size), Image.Resampling.BICUBIC))
    else:
        frames.append(vision_sized)

    values = []
    for frame in frames:
        arr = np.asarray(frame, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        values.append(np.transpose(arr, (2, 0, 1)))
    pixel_values = np.stack(values, axis=0)[None, ...].astype(np.float32)
    pixel_attention_mask = np.ones(
        (1, len(frames), max_image_size, max_image_size),
        dtype=np.bool_,
    )
    return pixel_values, pixel_attention_mask, rows, cols


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--atol", default=1e-4, type=float)
    args = parser.parse_args()

    pixel_values, pixel_attention_mask, rows, cols = build_image_inputs(args.image)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])

    full = session.run(
        ["image_features"],
        {
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask,
        },
    )[0]

    chunked_parts = []
    for idx in range(pixel_values.shape[1]):
        part = session.run(
            ["image_features"],
            {
                "pixel_values": pixel_values[:, idx : idx + 1, :, :, :],
                "pixel_attention_mask": pixel_attention_mask[:, idx : idx + 1, :, :],
            },
        )[0]
        chunked_parts.append(part)
    chunked = np.concatenate(chunked_parts, axis=0)

    diff = np.abs(full - chunked)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    shape_match = full.shape == chunked.shape
    pass_equivalence = shape_match and max_abs <= args.atol

    print(f"image={args.image}")
    print(f"model={args.model}")
    print(f"rows={rows} cols={cols} num_images={pixel_values.shape[1]}")
    print(f"pixel_values_shape={list(pixel_values.shape)}")
    print(f"full_shape={list(full.shape)}")
    print(f"chunked_shape={list(chunked.shape)}")
    print(f"max_abs_diff={max_abs:.10f}")
    print(f"mean_abs_diff={mean_abs:.10f}")
    print(f"CHUNKED_VISION_EQUIVALENCE: {'PASS' if pass_equivalence else 'FAIL'}")
    return 0 if pass_equivalence else 1


if __name__ == "__main__":
    raise SystemExit(main())
