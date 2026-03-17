#!/usr/bin/env python3
"""
tools/export_encoders.py — Export pretrained ONNX encoders for DendriteNet.

Produces two model files ready for use with -DDENDRITE_ONNX:
  models/mobilenet_v2_features.onnx  — image encoder, output [1, 1280]
  models/audio_features.onnx         — audio encoder,  output [1, 1024]

Usage:
  python tools/export_encoders.py               # export both
  python tools/export_encoders.py --image        # image only
  python tools/export_encoders.py --audio        # audio only
  python tools/export_encoders.py --out-dir /tmp/models

After export:
  # Linux / macOS
  export DENDRITE_IMAGE_MODEL=models/mobilenet_v2_features.onnx
  export DENDRITE_AUDIO_MODEL=models/audio_features.onnx

  # Windows (PowerShell)
  $Env:DENDRITE_IMAGE_MODEL = "models/mobilenet_v2_features.onnx"
  $Env:DENDRITE_AUDIO_MODEL = "models/audio_features.onnx"

  # Build with ONNX support
  g++ -O3 -std=c++17 -Iinclude -DDENDRITE_ONNX -ffast-math -funroll-loops ^
      -mavx2 -fopenmp -o dendrite3d examples/main.cpp ^
      -Ipath/to/onnxruntime/include -Lpath/to/onnxruntime/lib -lonnxruntime

Requirements:
  pip install torch torchvision
"""

import argparse
import os
import sys


# ---------------------------------------------------------------------------
# Image encoder: MobileNetV2 → 1280-dim feature embedding
# ---------------------------------------------------------------------------

def export_mobilenet_v2(out_dir: str) -> bool:
    """
    Export MobileNetV2 (ImageNet pretrained) penultimate features to ONNX.

    Input  node "input":  [1, 3, 224, 224] float32, NCHW,
                           ImageNet-normalised (mean=[0.485,0.456,0.406],
                                               std =[0.229,0.224,0.225]).
    Output node "output": [1, 1280] float32 — GlobalAvgPool features,
                           suitable as an image embedding.
    """
    try:
        import torch
        import torch.nn as nn
        import torchvision.models as tvm
    except ImportError:
        print("ERROR: torch and torchvision are required.")
        print("  Install: pip install torch torchvision")
        return False

    print("Exporting MobileNetV2 image encoder (pretrained on ImageNet) …")

    weights = tvm.MobileNet_V2_Weights.DEFAULT
    backbone = tvm.mobilenet_v2(weights=weights)

    class FeatureExtractor(nn.Module):
        """MobileNetV2 without the classifier — returns 1280-dim pool features."""
        def __init__(self, b: nn.Module) -> None:
            super().__init__()
            self.features = b.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.features(x)
            x = self.pool(x)
            return x.flatten(1)   # [batch, 1280]

    model = FeatureExtractor(backbone)
    model.eval()

    dummy = torch.zeros(1, 3, 224, 224)
    out_path = os.path.join(out_dir, "mobilenet_v2_features.onnx")

    # dynamo=False → stable TorchScript exporter (no onnxscript dep, no emoji crash on Windows)
    torch.onnx.export(
        model, dummy, out_path,
        dynamo=False,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    print(f"  Saved : {out_path}")
    print(f"  Input : [1, 3, 224, 224] float32  NCHW, ImageNet-normalised")
    print(f"  Output: [1, 1280] float32  GlobalAvgPool embedding")
    print(f"  Note  : Pre-process images to 224×224, normalise with ImageNet stats.")
    return True


# ---------------------------------------------------------------------------
# Audio encoder: 1D CNN → 1024-dim feature embedding
# ---------------------------------------------------------------------------

def export_audio_cnn(out_dir: str) -> bool:
    """
    Export a lightweight audio feature CNN to ONNX.

    Input  node "input":  [1, 15600] float32 — 0.975 s of 16 kHz mono audio,
                           amplitude range −1 … +1.
    Output node "output": [1, 1024] float32 — audio embedding.

    NOTE: This model is randomly initialised and serves as a structural
    placeholder.  For production accuracy, replace it with a pretrained model:

      Option A — PANNs (PyTorch Audio Neural Networks, strong performance):
        https://github.com/qiuqiangkong/audioset_tagging_cnn
        Download CNN14 weights and export the embedding layer.

      Option B — YAMNet (TensorFlow; needs tf2onnx):
        python -m tf2onnx.convert \\
            --saved-model path/to/yamnet_saved_model \\
            --output models/yamnet.onnx \\
            --opset 12
        Set DENDRITE_AUDIO_MODEL=models/yamnet.onnx and update
        input_shape to {1, 15600} in create_audio_module() if needed.

      Option C — Wav2Vec2 (HuggingFace transformers):
        from transformers import Wav2Vec2Model
        # export feature_extractor portion to ONNX …
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("ERROR: torch is required.")
        print("  Install: pip install torch")
        return False

    print("Exporting audio feature CNN (randomly initialised placeholder) …")

    class AudioFeatureCNN(nn.Module):
        """
        Lightweight 1D convolutional encoder.
        Input:  [batch, 15600]
        Output: [batch, 1024]

        Uses mean(dim=-1) for global temporal pooling — ONNX ReduceMean,
        exportable regardless of the exact time-axis length.
        BatchNorm omitted to keep the model self-contained and ONNX-friendly.
        """
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d( 1,   64, kernel_size=64, stride=8,  padding=28),
                nn.ReLU(inplace=True),
                nn.Conv1d( 64, 128, kernel_size=32, stride=4,  padding=14),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 256, kernel_size=16, stride=2,  padding=7),
                nn.ReLU(inplace=True),
                nn.Conv1d(256, 512, kernel_size=8,  stride=2,  padding=3),
                nn.ReLU(inplace=True),
            )
            self.fc = nn.Linear(512, 1024)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = x.unsqueeze(1)       # [batch, 1, 15600]
            x = self.conv(x)         # [batch, 512, T]
            x = x.mean(dim=-1)       # global avg pool → [batch, 512]  (ONNX ReduceMean)
            return self.fc(x)        # [batch, 1024]

    model = AudioFeatureCNN()
    model.eval()

    dummy = torch.zeros(1, 15600)
    out_path = os.path.join(out_dir, "audio_features.onnx")

    torch.onnx.export(
        model, dummy, out_path,
        dynamo=False,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    print(f"  Saved : {out_path}")
    print(f"  Input : [1, 15600] float32  (0.975 s mono 16kHz, range −1…+1)")
    print(f"  Output: [1, 1024] float32  audio embedding")
    print(f"  ⚠  Randomly initialised — replace with a pretrained model for")
    print(f"     real audio understanding (see docstring above).")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export encoder models to ONNX for DendriteNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--image",    action="store_true", help="Export MobileNetV2 image encoder")
    parser.add_argument("--audio",    action="store_true", help="Export audio feature CNN")
    parser.add_argument("--out-dir",  default="models",    help="Output directory (default: models/)")
    args = parser.parse_args()

    # Default: export both
    if not args.image and not args.audio:
        args.image = True
        args.audio = True

    os.makedirs(args.out_dir, exist_ok=True)

    results: list[tuple[str, bool]] = []
    if args.image:
        results.append(("image (MobileNetV2)", export_mobilenet_v2(args.out_dir)))
    if args.audio:
        results.append(("audio (CNN-1024)",    export_audio_cnn(args.out_dir)))

    print()
    all_ok = all(ok for _, ok in results)
    for label, ok in results:
        print(f"  {'✓' if ok else '✗'}  {label}")

    if all_ok:
        print()
        print("Models exported.  To enable ONNX encoders in DendriteNet:")
        print()
        if args.image:
            print(f"  export DENDRITE_IMAGE_MODEL={args.out_dir}/mobilenet_v2_features.onnx")
        if args.audio:
            print(f"  export DENDRITE_AUDIO_MODEL={args.out_dir}/audio_features.onnx")
        print()
        print("  g++ -O3 -std=c++17 -Iinclude -DDENDRITE_ONNX -ffast-math -funroll-loops \\")
        print("      -mavx2 -fopenmp -o dendrite3d examples/main.cpp \\")
        print("      -Ipath/to/onnxruntime/include -Lpath/to/onnxruntime/lib -lonnxruntime")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
