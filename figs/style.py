"""
Shared sizing/cropping for figs/generate_*.py, so new figures don't repeat the
whitespace and tiny-font mistakes reviewer comments kept flagging (R1.19,
R1.32, R1.41): every one of those turned out to be either (a) font sizes left
at Matplotlib's ~8-11pt default, illegible once shrunk to fit a two-column
page, or (b) dead margin left by `bbox_inches="tight"`, which computes an
inflated bounding box for 3D axes and doesn't account for post-hoc cropping
needs on 2D ones either.

Import the constants below instead of inventing new ones per script, and
call `crop_to_content` after `savefig` for any figure with a 3D axes or
otherwise-unreliable tight-bbox (dense scatter/surface plots).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Chosen so text reads clearly at the figure's final printed size (a
# \linewidth or \textwidth figure in a two-column IEEE paper, i.e. roughly
# 1/2 to 1 page width) -- not Matplotlib's defaults, which read fine on
# screen at the figure's native pixel size but shrink to near-illegible once
# placed in the paper. Tune per figure only if it's visibly cramped at this
# size; don't drop below these without a reason.
FS_TITLE = 16
FS_LABEL = 14
FS_TICK = 12
FS_NOTE = 11


def crop_to_content(path: str | Path, pad: int = 10, threshold: int = 250) -> None:
    """Crop a saved raster figure to its actual content, removing the dead
    margin `bbox_inches="tight"` leaves behind.

    For an image with an alpha channel (`savefig(..., transparent=True)`),
    crops to the non-transparent region -- this is the reliable fix for 3D
    axes, whose `get_tightbbox` includes a lot of empty space around the
    plotted surface regardless of view angle.

    For an opaque image, crops to content by treating all-but-near-white
    pixels as background; `threshold` (0-255) is how close to white (255)
    still counts as background. Use this when transparency isn't an option
    (e.g. multiple subplots sharing one background, like a subplot grid).
    """
    from PIL import Image

    img = Image.open(path)
    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    if has_alpha:
        arr = np.array(img.convert("RGBA"))
        mask = arr[:, :, 3] > 0
    else:
        arr = np.array(img.convert("RGB"))
        mask = (arr < threshold).any(axis=2)

    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return  # fully blank image: nothing sensible to crop to
    top, bottom = max(rows.min() - pad, 0), min(rows.max() + pad, arr.shape[0])
    left, right = max(cols.min() - pad, 0), min(cols.max() + pad, arr.shape[1])
    img.crop((left, top, right, bottom)).save(path)
