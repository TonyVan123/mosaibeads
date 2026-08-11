"""Developer utility: export the official pretrained MobileNetV3-Small feature encoder."""
from pathlib import Path

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.features = model.features
        self.pool = model.avgpool

    def forward(self, x):
        return torch.flatten(self.pool(self.features(x)), 1)


target = Path(__file__).resolve().parent / "semantic_encoder.onnx"
model = Encoder().eval()
torch.onnx.export(model, torch.zeros(1, 3, 224, 224), target,
                  input_names=["images"], output_names=["features"],
                  dynamic_axes={"images": {0: "batch"}, "features": {0: "batch"}},
                  opset_version=17, dynamo=False)
print(target)
