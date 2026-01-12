# EasyReach VLA

A simple Python CLI for running inference on OpenVLA vision-language-action models.

## What is OpenVLA?

OpenVLA is an open-source vision-language-action model for robotic manipulation. It takes language instructions and camera images as input and generates robot actions. The model was trained on 970K robot manipulation episodes from the Open X-Embodiment dataset.

## Setup

1.  **Install `uv`**:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Install Dependencies**:
    ```bash
    uv sync
    ```

## Usage

The CLI is built with `Typer` and defaults to `openvla/openvla-7b`.

### Basic Inference

```bash
uv run python inference.py run --instruction "Pick up the mug" --image-path ./test_image.png
```

### Mock Mode (No GPU/Model download required)

Test without downloading models:

```bash
uv run python inference.py run --instruction "Pick up the mug" --mock
```

### Available Models

- `openvla/openvla-7b` - Main model (7B parameters, default)
- `openvla/openvla-7b-finetuned-libero-spatial` - Fine-tuned for LIBERO tasks

### Using a Different Model

```bash
uv run python inference.py run \
  --instruction "Pick up the mug" \
  --image-path ./test_image.png \
  --model-id openvla/openvla-7b-finetuned-libero-spatial
```

### Options

- `--instruction`: (Required) The text instruction for the model.
- `--image-path`: Path to the input image. (Required unless in mock mode)
- `--model-type`: Type of model (default: `openvla`)
- `--model-id`: HuggingFace model ID (default: `openvla/openvla-7b`)
- `--device`: Device to run on: `cuda`, `cpu`, or `mps` (auto-detected by default)
- `--mock`: Run in mock mode without loading models

### Quick Examples

```bash
# OpenVLA 7B (default)
uv run python inference.py run --instruction "Pick up the mug" --image-path ./test_image.png

# Fine-tuned LIBERO model
uv run python inference.py run --instruction "Pick up the mug" --image-path ./test_image.png --model-id openvla/openvla-7b-finetuned-libero-spatial

# Mock mode (no model download)
uv run python inference.py run --instruction "Pick up the mug" --mock
```

## Model Downloading

Models will be **automatically downloaded** from HuggingFace Hub the first time you run inference without the `--mock` flag.

If you prefer to download manually:

```bash
uv run huggingface-cli download openvla/openvla-7b
```

## Troubleshooting

### Common Issues

- **Memory Issues**: OpenVLA 7B is a large model (~14GB). Make sure you have enough RAM/VRAM.
- **MPS on Mac**: The model runs on Apple Silicon Macs using MPS, but may be slower than CUDA.
- **NumPy/Protobuf Errors**: The dependencies are pinned to compatible versions. Run `uv sync` to ensure correct versions are installed.

### Dependency Versions

This project requires specific dependency versions for compatibility:

- `numpy>=1.21.0,<2.0.0` (TensorFlow compatibility)
- `protobuf>=3.20.0,<5.0.0` (TensorFlow compatibility)
- `timm>=0.9.10,<1.0.0` (OpenVLA requirement)

## Resources

- [OpenVLA GitHub](https://github.com/openvla/openvla)
- [OpenVLA Paper](https://arxiv.org/abs/2406.09246)
- [OpenVLA Website](https://openvla.github.io/)
- [HuggingFace Model Hub](https://huggingface.co/openvla)

## Version

EasyReach VLA v0.1.0
