import unittest
from unittest.mock import MagicMock, patch
from src.models.openvla import OpenVLAModel
from PIL import Image


class TestOpenVLAModel(unittest.TestCase):
    def setUp(self):
        self.model = OpenVLAModel()

    @patch("src.models.openvla.AutoProcessor")
    @patch("src.models.openvla.AutoModelForVision2Seq")
    def test_load_and_predict_flow(self, mock_model_cls, mock_processor_cls):
        # Setup mocks
        mock_processor_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model_instance
        mock_processor_cls.from_pretrained.return_value = mock_processor_instance

        # Test loading
        self.model.load_model("test-model-id")
        mock_model_cls.from_pretrained.assert_called()

        # Configure predict mock behavior
        mock_processor_instance.return_value = MagicMock()  # inputs
        mock_model_instance.dtype = "float32"
        mock_model_instance.generate.return_value = [1, 2, 3]  # fake token ids
        mock_processor_instance.batch_decode.return_value = ["Pick up the apple"]

        # Test predict
        img = Image.new("RGB", (100, 100))
        result = self.model.predict(img, "Instruction")

        self.assertEqual(result, "Pick up the apple")
        print("Mock inference test passed!")


if __name__ == "__main__":
    unittest.main()
