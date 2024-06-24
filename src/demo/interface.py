import gradio as gr
from gradio.utils import re
from DINO import Dino

classes = [
    "bitter pack",
    "bottle pack",
    "box",
    "can pack",
    "crate",
    "keg",
]

prompts = {
    "bitter pack": ["glass bottle", "bitter", "pack"],
    "bottle pack": ["glass bottle", "pack"],
    "box": ["box"],
    "can pack": ["can", "pack"],
    "crate": ["crate"],
    "keg": ["keg"],
}

try:
    model = Dino()
except Exception as e:
    raise e


def predict(image, progress=gr.Progress(track_tqdm=True)):
    progress(0, desc="Predicting...")
    outputs = model.predict_batch(image, prompts)
    results = postprocess(outputs, image)
    return image, results


def postprocess(output, image):
    """Postprocess for gr.AnnotatedImage
    Args:
    output: the output from the model (list[list[box], list[score], list[prompt]])
    Returns:
    results: a tuple of a base image and list of annotations (tuple[Image, list[Annotation]])
        where:
        Image: the base image (str filepath, numpy.ndarray, or PIL.Image)
        list[Annotation]: a list of annotations Annotation = tuple[box, label]
    """
    h, w, _ = image.shape
    annotations = []
    for key, value in output.items():
        for box, score, _ in zip(*value):
            # normalize the box and convert to int
            box = [int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)]
            annotations.append((box, f"{key} ({score})"))

    return annotations


with gr.Blocks() as demo:
    input = gr.Image()
    output = gr.AnnotatedImage()
    button = gr.Button(value="Predict")
    button.click(predict, inputs=[input], outputs=[output])


def run():
    demo.launch()


if __name__ == "__main__":
    run()
