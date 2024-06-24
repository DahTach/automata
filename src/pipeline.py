import json
from collections import defaultdict
import pathlib
import cv2 as cv
import torch
import torchvision
from typing import List, Tuple
from grounDet import Detector
from metrics import Metrics

json_sample = {
    "images": [
        {
            "name": "image1",
            "frame": "horizontal",
            "gr_truths": [("class_id", [0.1, 0.2, 0.3, 0.4])],
        },
    ],
    "aliases": [
        {
            "class_name": "class_id",
            "frame": "vertical",
            "alias": ["class_name"],
            "metrics": {
                "precision": 0.1,
                "recall": 0.2,
                "f1": 0.3,  # f1 score
            },
        }
    ],
}


class Pipeline:
    def __init__(self, dataset, inference, ground_truths):
        self.dataset = dataset
        self.inference = inference
        self.ground_truths = ground_truths
        self.vision = Detector("dino")
        self.lang = LLM()

    def run(self):
        images = [""]
        for image in images:
            image = Image(image)
            self.dataset.add_image(image)
        self.dataset.auto_alias()

    def auto_alias(self, class_id):
        alias = self.lang.generate(self.dataset.history)
        results = []
        metric = {
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
        }
        for image in self.dataset.images:
            result = self.vision.predict(image=image.path, prompt=alias)
            bench = self.vision.metrics(image.gr_truths, result)
            true_positives, false_positives, false_negatives = self.vision.metrics()
            metric["true_positive"] += true_positives
            metric["false_positive"] += false_positives
            metric["false_negative"] += false_negatives
            results.append(result)

        self.dataset.add_alias(
            Alias(
                class_id=class_id,
                class_name=self.dataset.classnames[class_id],
                alias=alias,
                metrics=Metrics(metric),
            )
        )
        self.dataset.save()
