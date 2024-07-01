from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import torch
import os
import numpy as np

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class PaliGemma:
    def __init__(self):
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            "google/paligemma-3b-mix-224",
            torch_dtype=torch.bfloat16,
            device_map="mps",
            revision="bfloat16",
        ).eval()
        self.processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-224")
        self.options = [
            "cardboard box",
            "plastic caps",
            "tin cans",
            "metal keg or gas canister",
            "crown corks",
        ]
        self.prompt = f"which of these categories: {self.options} better describes the object in the image?"

    def generate(self, image: np.ndarray):
        model_inputs = self.processor(
            text=self.prompt, images=image, return_tensors="pt"
        ).to(self.model.device)
        input_len = model_inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(
                **model_inputs, max_new_tokens=100, do_sample=False
            )
            generation = generation[0][input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded

    def crop_box(self, image: np.ndarray, box: list[int]):
        x1, y1, x2, y2 = box
        return image[y1:y2, x1:x2]
