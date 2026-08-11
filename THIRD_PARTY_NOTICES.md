# Third-party notices

## BeadColors palette data

Files under `assets/palettes/` are derived from the BeadColors project:

- Project: https://github.com/maxcleme/beadcolors
- Copyright: © 2020 maxcleme and palette contributors
- License: MIT

The palette RGB values are digital approximations. Real beads vary by production
batch, lighting, camera white balance, melting/ironing and display calibration.
For color-critical work, compare against a physical, ironed swatch card.

## Runtime libraries

MOSAIBEADS uses Python, NumPy, OpenCV and Pillow. Their respective
licenses are included by the packaged runtime or available from their projects.

## Optional MobileNetV3-Small model

`model_pack/semantic_encoder.onnx` is exported from the pretrained
`torchvision.models.mobilenet_v3_small` feature trunk.

- Project: https://github.com/pytorch/vision
- Documentation: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html
- Torchvision code license: BSD 3-Clause

The optional isolated GPU runtime uses ONNX Runtime and NVIDIA CUDA/cuDNN pip
runtime packages under their respective licenses. The base executable does not
bundle those large GPU libraries.
