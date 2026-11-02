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


def crop_pdf_to_content(path: str | Path, pad_pt: float = 8.0, threshold: int = 250,
                        dpi: int = 200) -> None:
    """Crop a saved vector PDF to its actual content, the PDF analogue of
    `crop_to_content` -- `bbox_inches="tight"` is unreliable for the same
    reason there (3D axes report an inflated bounding box), and there's no
    alpha channel to crop a vector page by, so this renders the page to a
    raster preview purely to measure the crop, then applies it to the PDF's
    own MediaBox (the page itself stays vector).

    Requires `pdftoppm` (poppler-utils) and the `pypdf` package.
    """
    import shutil
    import subprocess
    import tempfile

    from PIL import Image
    from pypdf import PdfReader, PdfWriter

    if shutil.which("pdftoppm") is None:
        raise RuntimeError("crop_pdf_to_content needs pdftoppm (poppler-utils) on PATH")

    path = Path(path)
    reader = PdfReader(path)
    page = reader.pages[0]
    box = page.mediabox
    page_w_pt, page_h_pt = float(box.width), float(box.height)

    with tempfile.TemporaryDirectory() as tmp:
        prefix = str(Path(tmp) / "preview")
        subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile", str(path), prefix],
                       check=True, capture_output=True)
        img = Image.open(prefix + ".png").convert("RGB")
        arr = np.array(img)

    mask = (arr < threshold).any(axis=2)
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return  # fully blank page: nothing sensible to crop to

    px_w, px_h = arr.shape[1], arr.shape[0]
    pt_per_px_x, pt_per_px_y = page_w_pt / px_w, page_h_pt / px_h
    left = max(cols.min() * pt_per_px_x - pad_pt, 0)
    right = min((cols.max() + 1) * pt_per_px_x + pad_pt, page_w_pt)
    # Image rows run top-to-bottom; PDF y runs bottom-to-top.
    top = max((px_h - rows.max() - 1) * pt_per_px_y - pad_pt, 0)
    bottom = min((px_h - rows.min()) * pt_per_px_y + pad_pt, page_h_pt)

    writer = PdfWriter()
    writer.append(reader)
    out_page = writer.pages[0]
    out_page.mediabox.lower_left = (left, top)
    out_page.mediabox.upper_right = (right, bottom)
    with open(path, "wb") as f:
        writer.write(f)
