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
    # "bitter pack": ["crown cork bottles"],
    "bottle pack": ["bottle pack", "water bottles", "water pack"],
    "box": ["cardboard box", "cardboard", "parcel"],
    "can pack": ["cans", "can pack", "tin pack", "tins", "tin cans", "aluminum cans", "beer cans"],
    "crate": ["plastic crate", "water crate"],
    "keg": ["keg", "beer keg", "alcohol keg", "metal keg", "canyster", "gas canyster"],
}

try:
    model = Dino()
except Exception as e:
    raise e


def predict(image, progress=gr.Progress(track_tqdm=True)):
    progress(0, desc="Predicting...")
    outputs = model.predict_slow(image, prompts)
    results = postprocess(outputs, image)
    return image, results


def postprocess(output, image):
    """
    Args:
        output: dict[str, Tuple[list[list[float]], list[float], list[str]]] = class predictions = (boxes, scores, labels)
    """
    h, w, _ = image.shape
    annotations = []
    for key, value in output.items():
        for box, score, _ in zip(*value):
            box = [int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)]
            annotations.append((box, f"{key} ({score})"))

    return annotations


with gr.Blocks() as demo:
    input = gr.Image()
    output = gr.AnnotatedImage()
    button = gr.Button(value="Predict")
    button.click(predict, inputs=[input], outputs=[output])


def run():
    demo.launch(share=True)


if __name__ == "__main__":
    run()
