# Model Architecture Figures

This directory contains the source code and assets for generating the architecture diagram of the ChromatogramCNN model.

## Files

- `plot_architecture.py`: Python script that uses `PlotNeuralNet` to generate the LaTeX TikZ code.
- `architecture_cnn.tex`: The generated LaTeX source file.
- `architecture_cnn.pdf`: The compiled vector graphic (PDF).
- `architecture_cnn.pdf.png`: The rendered raster image (PNG) for use in Markdown/Web.
- `init.tex`, `Box.sty`, `Ball.sty`, `RightBandedBox.sty`: TikZ style definitions required for compilation.

## Requirements

1. **Python 3**: To run the generator script.
2. **LaTeX (pdflatex)**: To compile the `.tex` file.
3. **ImageMagick (`convert`)** or **Poppler (`pdftoppm`)**: Optional, to convert PDF to PNG.

## Steps to Regenerate

### 1. Generate LaTeX Source
Run the Python script from the project root:
```bash
python3 figs/plot_architecture.py
```

### 2. Compile to PDF
Compile the generated LaTeX file using `pdflatex`. You must run this from the `figs/` directory so it can find the local `.sty` files:
```bash
cd figs
pdflatex architecture_cnn.tex
pdflatex architecture_cnn.tex  # Run twice for correct TikZ positioning
```

### 3. Convert to PNG (Optional)
```bash
# Using ImageMagick
convert -density 300 -trim architecture_cnn.pdf architecture_cnn.pdf.png

# Or using macOS 'sips'
sips -s format png architecture_cnn.pdf --out architecture_cnn.pdf.png
```
