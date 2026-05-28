#!/usr/bin/env python3
"""
PlotNeuralNet diagram for ChromatogramCNN.

Generates a LaTeX/TikZ 3-D block diagram of the full architecture:
  GC-MS input → Spectral Projection → 3× Dilated ResBlocks
  → Dual Pooling (Max ∥ Attention) → Concat → FC → 4 species

Usage:
  python figs/plot_architecture.py
  cd figs && pdflatex architecture_cnn.tex && pdflatex architecture_cnn.tex
  convert -density 300 -trim figs/architecture_cnn.pdf architecture_cnn.pdf.png
"""

import sys
import os

# ---- paths ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR   = os.path.dirname(SCRIPT_DIR)
PNN_DIR    = os.path.join(REPO_DIR, "PlotNeuralNet")
GCMS_IMG   = os.path.join(SCRIPT_DIR, "gc-ms.png")
OUT_TEX    = os.path.join(SCRIPT_DIR, "architecture_cnn.tex")

sys.path.insert(0, PNN_DIR)
from pycore.tikzeng import (
    to_head, to_cor, to_begin, to_end, to_generate,
    to_input, to_Conv, to_ConvConvRelu, to_Pool,
    to_SoftMax, to_Sum, to_connection,
)

# ---- inject raw LaTeX -------------------------------------------------------
def raw(s):
    return s

# ---- architecture -----------------------------------------------------------
arch = [
    # ---- preamble -----------------------------------------------------------
    raw(r"""
\documentclass[border=30pt, multi, tikz]{standalone}
\usepackage{import}
\subimport{""" + os.path.join(PNN_DIR, 'layers/').replace('\\', '/') + r"""}{init}
\usetikzlibrary{positioning,3d}
\usepackage{amsmath}
"""),
    to_cor(),

    # Custom color overrides and academic definitions
    raw(r"""
\definecolor{ResBlockFill}{HTML}{B7CFEA}
\definecolor{ResBlockEdge}{HTML}{1F4E9D}
\definecolor{PoolFill}{HTML}{F4B4B4}
\definecolor{PoolEdge}{HTML}{B22222}
\definecolor{LinearFill}{HTML}{C9E4C5}
\definecolor{LinearEdge}{HTML}{2E7D32}
\definecolor{InputFill}{HTML}{E0D4F0}
\definecolor{InputEdge}{HTML}{6A3D9A}

\def\ResColor{ResBlockFill}
\def\ResColorBand{ResBlockEdge}
\def\AttnColor{rgb:magenta,5;blue,3;white,6}
"""),

    to_begin(),

    # =========================================================================
    # 0.  GC-MS Input Image  [200 RT bins × 1000 m/z]
    # =========================================================================
    raw(r"""
\node[inner sep=2pt, draw=InputEdge, very thick,
      fill=white, rounded corners=3pt] (gcms) at (-7.0,0,0)
    {\includegraphics[width=7.5cm,height=7.5cm]{""" + GCMS_IMG + r"""}};
"""),
    raw(r"""
\node[font=\small\bfseries, above=3mm of gcms.north, align=center]
    {GC-MS Chromatogram};
\node[font=\scriptsize, below=4mm of gcms.south]
    {$200\,\text{RT bins} \times 1000\,m/z$};
"""),

    # =========================================================================
    # 1.  Per-bin spectral projection: Linear(1000 → 128)
    # =========================================================================
    to_Conv("proj", s_filer=" ", n_filer=" ",
            offset="(2,0,0)", to="(0,0,0)",
            height=40, depth=10, width=2,
            caption="{Spectral Proj.}"),
    raw(r"""\draw [connection] (gcms.east) -- node {\midarrow} (proj-west);"""),
    raw(r"""\node[font=\scriptsize, below=12pt] at (proj-south)
    {$200 \times 128$};"""),

    # =========================================================================
    # 2.  ResBlock — dilation 1   (receptive field: 7 bins)
    # =========================================================================
    # Redefine Conv colors to orange for ResBlocks
    raw(r"""
\def\ConvColor{\ResColor}
\def\ConvReluColor{\ResColorBand}
"""),

    to_ConvConvRelu("res1",
                    s_filer=" ", n_filer=(" ", " "),
                    offset="(2,0,0)", to="(proj-east)",
                    height=40, depth=10, width=(5, 2),
                    caption="{ResBlock (dilation=1)}"),
    to_connection("proj", "res1"),
    raw(r"""\node[font=\scriptsize, below=12pt] at ([xshift=-5pt]res1-south) {RF\,=\,7};"""),

    # =========================================================================
    # 3.  ResBlock — dilation 2   (receptive field: 19 bins)
    # =========================================================================
    to_ConvConvRelu("res2",
                    s_filer=" ", n_filer=(" ", " "),
                    offset="(2.5,0,0)", to="(res1-east)",
                    height=40, depth=10, width=(6, 2),
                    caption="{ResBlock (dilation=2)}"),
    to_connection("res1", "res2"),
    raw(r"""\node[font=\scriptsize, below=12pt] at ([xshift=-5pt]res2-south) {RF\,=\,19};"""),

    # =========================================================================
    # 4.  ResBlock — dilation 4   (receptive field: 43 bins)
    # =========================================================================
    to_ConvConvRelu("res3",
                    s_filer=" ", n_filer=(" ", " "),
                    offset="(2.5,0,0)", to="(res2-east)",
                    height=40, depth=10, width=(7, 2),
                    caption="{ResBlock (dilation=4)}"),
    to_connection("res2", "res3"),
    raw(r"""\node[font=\scriptsize, below=12pt] at ([xshift=-5pt]res3-south) {RF\,=\,43};"""),

    # Restore default colors for pools / head
    raw(r"""
\def\ConvColor{rgb:yellow,5;red,2.5;white,5}
\def\ConvReluColor{rgb:yellow,5;red,5;white,5}
"""),

    # =========================================================================
    # 5a. Global max-pool   → [128]   (upper branch)
    # =========================================================================
    to_Pool("maxpool",
            offset="(3.0, 7, 0)", to="(res3-east)",
            height=5, depth=10, width=2,
            caption="Max Pool"),
    raw(r"""\node[font=\scriptsize, above=2pt] at (maxpool-north) {$[128]$};"""),

    # =========================================================================
    # 5b. Soft attention-pool → [128]  (lower branch)
    # =========================================================================
    # Use SoftmaxColor for attention pool to distinguish it visually
    raw(r"""
\pic[shift={(3.0,-7,0)}] at (res3-east)
    {Box={
        name=attnpool,
        caption=Attn Pool,
        fill=\AttnColor,
        opacity=0.6,
        height=5,
        width=2,
        depth=10
    }};
\node[font=\scriptsize, below=2pt] at (attnpool-south) {$[128]$};
"""),

    # Fork: res3-east → maxpool-west  and  res3-east → attnpool-west
    raw(r"""
\draw [connection]
    (res3-east) -- ++(1.5,0,0) |- node[above,sloped,pos=0.75]{\midarrow} (maxpool-west);
\draw [connection]
    (res3-east) -- ++(1.5,0,0) |- node[below,sloped,pos=0.75]{\midarrow} (attnpool-west);
"""),

    # =========================================================================
    # 6.  Concatenation  [128 ∥ 128 → 256]
    # =========================================================================
    to_Sum("cat", offset="(5.5,0,0)", to="(res3-east)", radius=2.5, opacity=0.85),
    raw(r"""
\node[font=\small\bfseries, above=3mm] at (cat-north) {concat};
\node[font=\scriptsize,     below=4mm] at (cat-south)  {\textbf{[256]}};
"""),

    # Pool branches merge into concat
    raw(r"""
\draw [connection]
    (maxpool-east) -| node[above,sloped,pos=0.3]{\midarrow} (cat-north);
\draw [connection]
    (attnpool-east) -| node[below,sloped,pos=0.3]{\midarrow} (cat-south);
"""),

    # =========================================================================
    # 7.  FC classification head: Linear(256 → 4)
    # =========================================================================
    to_SoftMax("fc", s_filer=4,
               offset="(2.5,0,0)", to="(cat-east)",
               height=5, depth=10, width=2,
               opacity=0.9, caption="FC"),
    to_connection("cat", "fc"),
    raw(r"""\node[font=\small\bfseries, right=4mm] at (fc-east)
    {\begin{tabular}{l}SNA\\GUR\\TAR\\BCO\end{tabular}};"""),

    to_end(),
]

# ---- generate ---------------------------------------------------------------
if __name__ == "__main__":
    to_generate(arch, OUT_TEX)
    print(f"Generated: {OUT_TEX}")
    print()
    print("Compile with:")
    print(f"  cd figs && pdflatex architecture_cnn.tex && pdflatex architecture_cnn.tex")
    print()
    print("Convert to PNG (requires ImageMagick):")
    print(f"  convert -density 300 -trim figs/architecture_cnn.pdf architecture_cnn.pdf.png")
    print()
    print("Or (requires poppler):")
    print(f"  pdftoppm -r 300 figs/architecture_cnn.pdf arch && convert arch-1.ppm architecture_cnn.pdf.png")
