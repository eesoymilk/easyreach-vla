from abc import ABC, abstractmethod
from PIL import Image


class VLAModel(ABC):
    """Abstract base class for Vision-Language-Action models."""

    @abstractmethod
    def load_model(self, model_id: str, **kwargs):
        """Loads the model and processor/tokenizer."""
        pass

    @abstractmethod
    def predict(self, image: Image.Image, instruction: str, **kwargs) -> str:
        """
        Runs inference on the image and instruction.

        Args:
            image: PIL Image object.
            instruction: Text instruction string.
            **kwargs: Additional inference parameters.

        Returns:
            Generated action or text response.
        """
        pass
