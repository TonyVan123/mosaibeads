# Quality verification

## V2.0 verification — 2026-08-11

- Unit/integration tests: 9 passed, including exact fixed-palette enforcement,
  click-color ranking and three-scheme auto tuning.
- Portable executable: `BeadSketchStudio_v2.0.exe`, 67,893,297 bytes.
- Packaged smoke test: exit code 0 from the delivered E-drive executable.
- EXE SHA-256: `70acb9177b2cdcb84ecc771bce85d8c4f1a98f4d3fef2eb1d79cd7172168abb0`.
- AI model: 3,717,007 bytes; SHA-256
  `f405388cb07a6eaaa439dcb5d384d48a5c758c9a48e4a5b6a945cfa4363479a5`.
- GPU verification: a real MobileNetV3 batch inference completed with
  `CUDAExecutionProvider` on the NVIDIA GeForce RTX 4060 Laptop GPU.
- CPU fallback: the same ONNX model was loaded and scored through OpenCV DNN.
- Auto-tune end-to-end: nine preview candidates plus three final schemes completed
  with GPU semantic scoring; provider metadata was propagated to the result.

The optional CUDA/cuDNN environment is intentionally not bundled into the base
EXE or compact model pack because it is approximately 3GB. The included installer
creates it beside the model on demand.

## Automated checks

`python -m unittest discover -s tests -v` verifies:

- sRGB ↔ Lab reference values;
- the published CIEDE2000 Sharma test pair (expected ΔE 2.0425);
- all four packaged palette files;
- output dimensions, color limit and bead counts;
- thin salient-line rescue at a 10:1 downscale;
- creation of all six export artifacts, including PNG, PDF, CSV and JSON.

## Visual test set

Development comparisons used the standard `skimage.data` astronaut portrait, cat,
coffee and rocket-pad photographs, plus synthetic line art. They are not distributed
with the application.

The important finding was that an early, aggressively dominant-color version looked
clean but removed too much mid-frequency facial and photographic detail. The final
sampler therefore blends a stable area reference with a structure-aware representative:

- low-saliency flat cells stay close to the stable reference;
- edge/face/saliency cells spend more of their weight on a real source-color medoid;
- a coherent high-contrast minority can be rescued when a thin feature occupies too
  little of a cell to win ordinary averaging;
- spatial cleanup is weakened at real edges.

On a synthetic 240×240 image containing a 3 px diagonal black line, conversion to
24×24 kept 22 dark line cells in the recorded development run, while the box-average
palette baseline produced no cells below the same luminance threshold. This test is
also retained as a less brittle minimum check (`>=16` cells) in the test suite.

## Practical acceptance criteria

- Default 48-bead portrait generation completes in a few seconds on the development PC.
- A result always uses no more than the selected color limit.
- The editor can repaint any cell and undo/redo the change.
- The material count equals width × height after editing.
- Exported charts contain cell codes, 5-cell guides, heavier 29×29 board boundaries and
  a per-color material legend.
