import typer
import sys
from PIL import Image
from src.factory import ModelFactory
from unittest.mock import MagicMock
from typing import Optional

app = typer.Typer()


@app.command(name="run")
def run(
    instruction: str = typer.Option(..., help="Instruction for the model."),
    model_type: str = typer.Option(
        "openvla", help="Type of model to use (default: openvla)."
    ),
    model_id: Optional[str] = typer.Option(
        None, help="HuggingFace model ID (default: openvla/openvla-7b)."
    ),
    image_path: Optional[str] = typer.Option(None, help="Path to the input image."),
    device: Optional[str] = typer.Option(
        None, help="Device to run on (cuda, cpu, mps)."
    ),
    mock: bool = typer.Option(False, help="Run in mock mode (no model loading)."),
):
    """
    Run inference on OpenVLA models.
    """
    try:
        # Set default model_id if not provided
        if model_id is None:
            model_id = "openvla/openvla-7b"

        print(f"Using model: {model_id} (type: {model_type})")

        if mock:
            print("Running in MOCK mode.")
            model = ModelFactory.create_model(model_type)
            # Mocking the internal methods
            model.load_model = MagicMock()
            model.predict = MagicMock(return_value="Mock action: Pick up object")
        else:
            model = ModelFactory.create_model(model_type)

        # Load model
        model.load_model(model_id, device=device)

        # Load image
        if mock and not image_path:
            image = Image.new("RGB", (224, 224), color="red")
        elif image_path:
            image = Image.open(image_path).convert("RGB")
        else:
            print(
                "Error: --image_path is required unless in --mock mode with default mock image generation."
            )
            sys.exit(1)

        # Inference
        print(f"Instruction: {instruction}")
        result = model.predict(image, instruction)
        print(f"Result: {result}")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


@app.command()
def version():
    """Print the version."""
    print("OpenVLA Inference v0.1.0")


if __name__ == "__main__":
    app()
