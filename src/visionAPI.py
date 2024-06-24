import gradio as gr
from DINO import Dino
from metrics import confusion_matrix
import os
from dataset import Dataset
from tqdm import tqdm
import time
from typing import Any
import numpy as np
from collections import defaultdict

# TODO: try DINO 1.5 PRO https://deepdataspace.com/playground/grounding_dino


cwd = os.getcwd()

SAMPLE_DATASET_PATH = os.path.join(cwd, "dataset_sample")

# dataset = Dataset("/Users/francescotacinelli/Developer/datasets/pallets_sorted/labeled/images/")

try:
    DATASET_PATH = os.getenv("DATASET_PATH") or SAMPLE_DATASET_PATH
    if not DATASET_PATH:
        raise ValueError("DATASET_PATH is not set")
    dataset = Dataset(DATASET_PATH)
    dataset.populate()
except Exception as e:
    raise e

try:
    model = Dino()
except Exception as e:
    raise e


def predict(
    alias,
    class_id,
    box_threshold=0.1,
    text_threshold=0.1,
    progress=gr.Progress(),
):
    progress(0, desc="Predicting...")
    total = len(dataset.images)

    metrics = [0, 0, 0]
    annotation = None
    for i, image in enumerate(tqdm(dataset.images)):
        img_bgr = image.data
        result = model.predict(img_bgr, alias, box_threshold, text_threshold)

        # calculate the confusion matrix for each confidence level
        metric = confusion_matrix(
            predictions=result,
            ground_truths=image.grounds,
            device=model.device,
            class_id=class_id,
        )

        metrics[0] += metric[0]
        metrics[1] += metric[1]
        metrics[2] += metric[2]

        precision = metrics[0] / (metrics[0] + metrics[1]) or 0
        recall = metrics[0] / (metrics[0] + metrics[2]) or 0

        # every 10 images, update the draw image
        if i % 10 == 0:
            annotation = annotations(result, img_bgr, dataset.classnames[class_id])

        yield (
            {f"progress {i}/{total}": i / total},
            {"precision": precision, "recall": recall},
            annotation,
        )


def annotations(boxes, image, classname):
    """
    Expects a a tuple of a base image and list of annotations: a tuple[Image, list[Annotation]].
    The Image itself can be str filepath, numpy.ndarray, or PIL.Image. Each Annotation is a tuple[Mask, str].
    The Mask can be either a tuple of 4 int's representing the bounding box coordinates (x1, y1, x2, y2), or 0-1 confidence mask in the form of a numpy.ndarray of the same shape as the image, while the second element of the Annotation tuple is a str label.
    """
    h, w, _ = image.shape

    annotations = []
    for box in boxes:
        box = int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)
        annotations.append((box, classname))

    return image, annotations


detector = gr.Interface(
    predict,
    inputs=[
        gr.Textbox(),
        gr.Dropdown(
            label="Class ID",
            choices=dataset.classnames,
            type="index",
        ),
        gr.Slider(minimum=0, maximum=1.0, step=0.01, value=0.1),
        gr.Slider(minimum=0, maximum=1.0, step=0.01, value=0.1),
    ],
    outputs=[
        gr.Label(label=None),
        gr.Label(
            label=None,
            value={"precision (allucinations)": 0, "recall (unrecognized)": 0},
        ),
        gr.AnnotatedImage(),
    ],
    api_name="predict",
)


def run():
    detector.launch(allowed_paths=[DATASET_PATH], share=True, show_error=True)


if __name__ == "__main__":
    run()
