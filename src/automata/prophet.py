import torch
import torchvision.ops as ops
import operations.nms as detops
import numpy as np
from models.dino import Dino
from models.pgemma import PaliGemma
from models.fastsam import Segmenter
import re
from prompts import Dataset
import os

ds = Dataset()


def get_device():
    if torch.cuda.is_available():
        print("using cuda")
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("using mps but mps is not fully supported yet")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        return "mps"
    else:
        print("using cpu")
        return "cpu"


class Prophet:
    def __init__(self, models: list[str]):
        self.prompts = ds.captions
        self.device = get_device()
        self.load(models)

    def load(self, models: list[str]):
        for model in models:
            if model == "oracle":
                self.oracle = PaliGemma()
                self.options = self.oracle.options
            elif model == "dino":
                self.dino = Dino(size="small")
            elif model == "fastsam":
                self.sam = Segmenter()
            else:
                raise ValueError(f"Model {model} not found")

    def predict(self, image: np.ndarray):
        detections = self.dino.predict(image, self.prompts)

        prophesies, comparison = self.prophesy(image, detections)

        return detections, prophesies, comparison

    def pred_seg(self, image: np.ndarray):
        predictions = self.dino.predict(image, self.prompts)

        preds_with_masks = {
            id: (
                torch.tensor([]).to(self.device),
                torch.tensor([]).to(self.device),
                torch.tensor([]).to(self.device),
            )
            for id in self.prompts.keys()
        }

        for class_id, (boxes, scores) in predictions.items():
            for idx in range(boxes.shape[0]):
                box = boxes[idx]
                score = scores[idx]
                res = self.sam.infer(image)

                print("masks = ", mask)
                # preds_with_masks[class_id] = (
                #     torch.cat((predictions[class_id][0], box.view(1, -1))),
                #     torch.cat((predictions[class_id][0], box.view(1, -1))),
                #     torch.cat((predictions[class_id][1], score.view(1))),
                # )

        return preds_with_masks

    def prophecy_id(self, prophecy: str):
        prophecy = re.sub(r"[^a-zA-Z0-9\s]", "", prophecy).lower().strip()

        for key, val in self.options.items():
            if prophecy in key:
                return val
        return len(self.options)

    def prophesy(
        self,
        image: np.ndarray,
        cleaned_predictions: dict[int, tuple[torch.Tensor, torch.Tensor]],
    ):
        prophesies = {
            i: (torch.tensor([]).to(self.device), torch.tensor([]).to(self.device))
            for i in range(len(self.options.keys()) + 1)
        }

        comparison = {i: [] for i in range(len(self.options.keys()))}

        for key, (boxes, scores) in cleaned_predictions.items():
            for idx in range(boxes.shape[0]):
                box = boxes[idx]
                score = scores[idx]

                img = self.oracle.crop_box(image, [int(i) for i in box])
                prophecy = self.oracle.generate(img)
                class_id = self.prophecy_id(prophecy)

                comparison[key].append(class_id)

                prophesies[class_id] = (
                    torch.cat((prophesies[class_id][0], box.view(1, -1))),
                    torch.cat((prophesies[class_id][1], score.view(1))),
                )

        return prophesies, comparison
