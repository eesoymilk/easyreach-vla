from src.interfaces import VLAModel
from src.models.openvla import OpenVLAModel


class ModelFactory:
    """Factory to create VLA models."""

    @staticmethod
    def create_model(model_type: str) -> VLAModel:
        model_type = model_type.lower()
        if model_type == "openvla":
            return OpenVLAModel()
        else:
            raise ValueError(
                f"Unknown model type: {model_type}. Only 'openvla' is supported."
            )
