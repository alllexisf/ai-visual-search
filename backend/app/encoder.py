import threading
import numpy as np
from PIL import Image
from .config import Settings


class ClipEncoder:
    """Exactly the same CPU encoder and normalization at build and query time."""
    def __init__(self, settings: Settings):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        torch.set_num_threads(settings.torch_threads)
        self.torch = torch
        self.lock = threading.Lock()
        self.processor = CLIPProcessor.from_pretrained(settings.model_id, revision=settings.model_revision)
        self.model = CLIPModel.from_pretrained(settings.model_id, revision=settings.model_revision).to('cpu').eval()

    def images(self, images: list[Image.Image]) -> np.ndarray:
        with self.lock, self.torch.inference_mode():
            inputs = self.processor(images=images, return_tensors='pt')
            features = self.model.get_image_features(**inputs)
            return self._normalize(features)

    def text(self, query: str) -> np.ndarray:
        with self.lock, self.torch.inference_mode():
            inputs = self.processor(text=[query], return_tensors='pt', padding=True, truncation=True, max_length=77)
            return self._normalize(self.model.get_text_features(**inputs))[0]

    @staticmethod
    def _normalize(features) -> np.ndarray:
        vectors = features.float().cpu().numpy()
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
