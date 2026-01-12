import torch
import numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
from src.interfaces import VLAModel


class OpenVLAModel(VLAModel):
    """OpenVLA model implementation."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # For Mac MPS support
        if torch.backends.mps.is_available():
            self.device = "mps"

    def load_model(self, model_id: str, **kwargs):
        """
        Loads the OpenVLA model and processor.

        Args:
            model_id: HuggingFace model ID.
            **kwargs: Extra arguments for from_pretrained (e.g., torch_dtype, low_cpu_mem_usage).
        """
        # Handle 'device' argument if passed in kwargs, but don't pass it to from_pretrained
        if "device" in kwargs:
            if kwargs["device"]:
                self.device = kwargs["device"]
            del kwargs["device"]

        print(f"Loading OpenVLA model: {model_id} on {self.device}...")

        # Default kwargs for OpenVLA if not provided
        # Note: OpenVLA is massive, so often requires 4-bit quantization or similar if running on smaller GPUs across generic hardware
        # But here we stick to standard loading unless specified.

        # We need trust_remote_code=True for OpenVLA as it often has custom modeling code
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if self.device != "cpu" else torch.float32,
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager",  # Disable SDPA to avoid _supports_sdpa error
        }
        load_kwargs.update(kwargs)

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(model_id, **load_kwargs).to(
            self.device
        )

        print("Model loaded successfully.")

    def predict(self, image: Image.Image, instruction: str, **kwargs) -> str:
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # OpenVLA prompt structure
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

        # Process inputs
        inputs = self.processor(prompt, image).to(self.device, dtype=self.model.dtype)

        # Get action prediction
        # unnorm_key specifies the dataset statistics to use for denormalization
        unnorm_key = kwargs.get("unnorm_key", "bridge_orig")
        do_sample = kwargs.get("do_sample", False)

        with torch.inference_mode():
            # Use predict_action() method which returns continuous action vectors
            action = self.model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=do_sample)

        # Format action output as readable string
        # Action shape is typically (7,) for 7-DoF: [x, y, z, roll, pitch, yaw, gripper]
        # Convert to numpy if it's a tensor, otherwise use as-is
        if isinstance(action, torch.Tensor):
            action_np = action.cpu().numpy()
        else:
            action_np = np.array(action) if not isinstance(action, np.ndarray) else action

        result = f"Predicted Action (7-DoF):\n"
        result += f"  Position (x, y, z): [{action_np[0]:.4f}, {action_np[1]:.4f}, {action_np[2]:.4f}]\n"
        result += f"  Rotation (r, p, y): [{action_np[3]:.4f}, {action_np[4]:.4f}, {action_np[5]:.4f}]\n"
        result += f"  Gripper: {action_np[6]:.4f}\n"
        result += f"\nRaw action: {action_np}"

        return result
