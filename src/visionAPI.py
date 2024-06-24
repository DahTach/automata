import gradio as gr
from DINO import Dino
from metrics import confusion_matrix
import os
from dataset import Dataset
from tqdm import tqdm
import time
from typing import Any


# dataset = Dataset("/Users/francescotacinelli/Developer/datasets/pallets_sorted/labeled/images/")

try:
    DATASET_PATH = (
        os.getenv("DATASET_PATH")
        or "/Users/francescotacinelli/Developer/datasets/pallets_sorted/labeled/images/"
    )
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


def predict(alias, box_threshold, text_threshold, progress=gr.Progress()):
    progress(0, desc="Predicting...")

    total = len(dataset.images)
    results = []
    metrics = []
    for i, image in enumerate(tqdm(dataset.images)):
        # Predict
        img_bgr = image.data
        result = model.predict(img_bgr, alias, box_threshold, text_threshold)
        # Calculate the metrics
        metric = confusion_matrix(predictions=result, ground_truths=image.grounds)

        # results.append(result.tolist())
        metrics.append(metric)
        progress = f"{i}/{total}"
        yield progress, metrics


result_samples: list[list[Any]] = [
    [(1, 2, 3), (4, 5, 6)],
]

detector = gr.Interface(
    predict,
    inputs=[
        gr.Textbox(),
        gr.Slider(minimum=0.1, maximum=1.0, step=0.01, value=0.5),
        gr.Slider(minimum=0.1, maximum=1.0, step=0.01, value=0.5),
    ],
    outputs=[
        gr.Textbox(label="Progress"),
        gr.DataFrame(
            headers=[
                "true_positives",
                "false_positives",
                "false_negatives",
            ],
            label="Results",
        ),
    ],
    api_name="predict",
)


def run():
    detector.launch(allowed_paths=[DATASET_PATH])


if __name__ == "__main__":
    run()
