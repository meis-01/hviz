from __future__ import annotations

import argparse
from pathlib import Path
import struct
import sys
import zlib

import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import hviz


def identity(q: NDArray[np.float64]) -> NDArray[np.float64]:
    return q


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hviz identity demo assets.")
    parser.add_argument(
        "--png-dir",
        type=Path,
        default=PROJECT_ROOT / "docs" / "images",
        help="Directory for README PNG images.",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Also write interactive HoloViews HTML examples under docs/examples.",
    )
    args = parser.parse_args()

    args.png_dir.mkdir(parents=True, exist_ok=True)
    images = {
        "identity-slice.png": render_slice_image(),
        "identity-components.png": render_components_image(),
        "identity-magnitude-phase.png": render_magnitude_phase_image(),
        "identity-surface.png": render_surface_image(),
    }
    for filename, image in images.items():
        path = args.png_dir / filename
        write_png(path, image)
        print(path.relative_to(PROJECT_ROOT))

    if args.html:
        html_dir = PROJECT_ROOT / "docs" / "examples"
        html_dir.mkdir(parents=True, exist_ok=True)
        examples = [
            (
                "identity-slice.html",
                hviz.plot_slice(identity, (-3.0, 3.0, 360), (-3.0, 3.0, 360), title="q", width=560, height=560),
                "bokeh",
            ),
            (
                "identity-components.html",
                hviz.plot_components(identity, (-3.0, 3.0, 240), (-3.0, 3.0, 240), title="q"),
                "bokeh",
            ),
            (
                "identity-magnitude-phase.html",
                hviz.plot_magnitude_phase(identity, (-3.0, 3.0, 240), (0.0, 3.0, 240), title="q"),
                "bokeh",
            ),
            (
                "identity-surface.html",
                hviz.plot_surface(identity, (-3.0, 3.0, 120), (0.0, 3.0, 120), title="||q||"),
                "plotly",
            ),
        ]
        for filename, plot, backend in examples:
            path = html_dir / filename
            hviz.save(plot, path, backend=backend)
            print(path.relative_to(PROJECT_ROOT))


def render_slice_image() -> NDArray[np.uint8]:
    grid = hviz.sample_slice(identity, (-3.0, 3.0, 560), (-3.0, 3.0, 560))
    return rgba_to_rgb(np.flipud(grid.image))


def render_magnitude_phase_image() -> NDArray[np.uint8]:
    grid = hviz.sample_re_im_norm(identity, (-3.0, 3.0, 360), (0.0, 3.0, 360))
    magnitude = hviz.quaternion_norm(grid.values)
    magnitude_rgb = apply_palette(magnitude, VIRIDIS)
    phase_rgb = rgba_to_rgb(np.flipud(grid.phase_image))
    return hstack_with_gap([np.flipud(magnitude_rgb), phase_rgb], gap=18)


def render_components_image() -> NDArray[np.uint8]:
    grid = hviz.sample_slice(identity, (-3.0, 3.0, 280), (-3.0, 3.0, 280), imag_axis="equal")
    panels = [diverging_color(grid.values[..., index]) for index in range(4)]
    top = hstack_with_gap([np.flipud(panels[0]), np.flipud(panels[1])], gap=14)
    bottom = hstack_with_gap([np.flipud(panels[2]), np.flipud(panels[3])], gap=14)
    return vstack_with_gap([top, bottom], gap=14)


def render_surface_image() -> NDArray[np.uint8]:
    grid = hviz.sample_re_im_norm(identity, (-3.0, 3.0, 130), (0.0, 3.0, 130))
    x, y = np.meshgrid(grid.real, grid.imag_norm)
    z = hviz.quaternion_norm(grid.values)
    return point_surface_thumbnail(x, y, z, width=720, height=520)


def rgba_to_rgb(rgba: NDArray[np.float64]) -> NDArray[np.uint8]:
    alpha = rgba[..., 3:4]
    rgb = rgba[..., :3] * alpha + (1.0 - alpha)
    return np.rint(np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def apply_palette(values: NDArray[np.float64], palette: NDArray[np.float64]) -> NDArray[np.uint8]:
    finite = values[np.isfinite(values)]
    low = float(np.nanmin(finite)) if finite.size else 0.0
    high = float(np.nanmax(finite)) if finite.size else 1.0
    if high == low:
        high = low + 1.0
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    positions = scaled * (len(palette) - 1)
    lower = np.floor(positions).astype(int)
    upper = np.clip(lower + 1, 0, len(palette) - 1)
    amount = positions - lower
    rgb = palette[lower] * (1.0 - amount[..., np.newaxis]) + palette[upper] * amount[..., np.newaxis]
    return np.rint(np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def diverging_color(values: NDArray[np.float64]) -> NDArray[np.uint8]:
    finite = values[np.isfinite(values)]
    limit = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    if limit == 0.0:
        limit = 1.0
    scaled = np.clip((values / limit + 1.0) / 2.0, 0.0, 1.0)
    return apply_palette(scaled, DIVERGING)


def point_surface_thumbnail(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    *,
    width: int,
    height: int,
) -> NDArray[np.uint8]:
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    x_scaled = (x - np.min(x)) / (np.max(x) - np.min(x))
    y_scaled = (y - np.min(y)) / (np.max(y) - np.min(y))
    z_scaled = (z - np.min(z)) / (np.max(z) - np.min(z))

    u = x_scaled - 0.82 * y_scaled
    v = 0.35 * x_scaled + 0.45 * y_scaled - 0.80 * z_scaled
    u = (u - np.min(u)) / (np.max(u) - np.min(u))
    v = (v - np.min(v)) / (np.max(v) - np.min(v))
    px = np.rint(70 + u * (width - 140)).astype(int)
    py = np.rint(45 + v * (height - 90)).astype(int)
    colors = apply_palette(z, VIRIDIS)

    order = np.argsort((x_scaled + y_scaled).ravel())
    flat_x = px.ravel()
    flat_y = py.ravel()
    flat_colors = colors.reshape(-1, 3)
    for index in order:
        draw_disc(image, int(flat_x[index]), int(flat_y[index]), 2, flat_colors[index])
    return image


def draw_disc(image: NDArray[np.uint8], cx: int, cy: int, radius: int, color: NDArray[np.uint8]) -> None:
    height, width, _ = image.shape
    y0 = max(0, cy - radius)
    y1 = min(height - 1, cy + radius)
    x0 = max(0, cx - radius)
    x1 = min(width - 1, cx + radius)
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius * radius:
                image[y, x] = color


def hstack_with_gap(images: list[NDArray[np.uint8]], *, gap: int) -> NDArray[np.uint8]:
    height = max(image.shape[0] for image in images)
    normalized = [pad_to_height(image, height) for image in images]
    spacer = np.full((height, gap, 3), 255, dtype=np.uint8)
    out = normalized[0]
    for image in normalized[1:]:
        out = np.concatenate([out, spacer, image], axis=1)
    return out


def vstack_with_gap(images: list[NDArray[np.uint8]], *, gap: int) -> NDArray[np.uint8]:
    width = max(image.shape[1] for image in images)
    normalized = [pad_to_width(image, width) for image in images]
    spacer = np.full((gap, width, 3), 255, dtype=np.uint8)
    out = normalized[0]
    for image in normalized[1:]:
        out = np.concatenate([out, spacer, image], axis=0)
    return out


def pad_to_height(image: NDArray[np.uint8], height: int) -> NDArray[np.uint8]:
    if image.shape[0] == height:
        return image
    pad = np.full((height - image.shape[0], image.shape[1], 3), 255, dtype=np.uint8)
    return np.concatenate([image, pad], axis=0)


def pad_to_width(image: NDArray[np.uint8], width: int) -> NDArray[np.uint8]:
    if image.shape[1] == width:
        return image
    pad = np.full((image.shape[0], width - image.shape[1], 3), 255, dtype=np.uint8)
    return np.concatenate([image, pad], axis=1)


def write_png(path: Path, rgb: NDArray[np.uint8]) -> None:
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("PNG input must have shape (height, width, 3)")

    height, width, _ = rgb.shape
    scanlines = b"".join(b"\x00" + row.tobytes() for row in rgb)
    chunks = [
        _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
        _png_chunk(b"IDAT", zlib.compress(scanlines, level=9)),
        _png_chunk(b"IEND", b""),
    ]
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"".join(chunks))


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


VIRIDIS = np.array(
    [
        [68, 1, 84],
        [59, 82, 139],
        [33, 145, 140],
        [94, 201, 97],
        [253, 231, 37],
    ],
    dtype=np.float64,
) / 255.0

DIVERGING = np.array(
    [
        [49, 54, 149],
        [69, 117, 180],
        [224, 243, 248],
        [255, 255, 255],
        [254, 224, 144],
        [215, 48, 39],
        [165, 0, 38],
    ],
    dtype=np.float64,
) / 255.0


if __name__ == "__main__":
    main()
