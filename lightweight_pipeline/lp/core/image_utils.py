from __future__ import annotations

import io
import subprocess
from pathlib import Path

from PIL import Image


def ensure_png(path: Path, heif_bin: str = "heif-converter") -> Path:
    ext = path.suffix.lower()
    if ext in {".png"}:
        return path
    if ext in {".jpg", ".jpeg"}:
        out = path.with_suffix(".png")
        Image.open(path).convert("RGB").save(out, format="PNG")
        return out
    if ext in {".heif", ".heic"}:
        out = path.with_suffix(".png")
        _heif_to_png(path, out, heif_bin)
        return out
    raise ValueError(f"unsupported image format: {path}")


def _heif_to_png(src: Path, dst: Path, heif_bin: str) -> None:
    try:
        subprocess.run(
            [heif_bin, str(src), str(dst)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"未找到 {heif_bin}，请安装 heif-converter 或在 config.yaml 中修改 entry.heif_converter_bin"
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"heif-converter 失败: {e.stderr}") from e


def load_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def crop_image(src: Path, bbox: tuple[int, int, int, int], dst: Path) -> Path:
    x0, y0, x1, y1 = bbox
    with Image.open(src) as im:
        cropped = im.crop((x0, y0, x1, y1))
        dst.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(dst, format="PNG")
    return dst


def pad_image_ratio(src: Path, ratio: float, dst: Path) -> Path:
    with Image.open(src) as im:
        w, h = im.size
        pad_w = int(w * ratio)
        pad_h = int(h * ratio)
        canvas = Image.new(im.mode, (w + 2 * pad_w, h + 2 * pad_h), color=(255, 255, 255))
        canvas.paste(im, (pad_w, pad_h))
        dst.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(dst, format="PNG")
    return dst
