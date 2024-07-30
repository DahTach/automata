import gradio as gr
from automata.prophet import Prophet
import numpy as np
import gradiops.ops as grops
from prompts import Dataset

ds = Dataset()


class Interface:
    def __init__(self):
        self.prophet = None
        self.interface()

    def load_model(self, progress=gr.Progress(track_tqdm=True)):
        progress(0, "Loading models...", 100)
        self.prophet = Prophet(models=["dino", "oracle"])
        print("Models loaded!")

    def predict(self, image: np.ndarray):
        prophecies = self.prophet.predict(image)
        prophecies_annotations = grops.to_annotations(prophecies, ds.legend)
        return image, prophecies_annotations

    def predict_compare(self, image: np.ndarray):
        detections, prophecies, comparison = self.prophet.predict_compare(image)

        prophecies_annotations = grops.to_annotations(prophecies, ds.legend)
        comparison_dataframe = grops.to_dataframe(comparison, ds.legend)

        return (image, prophecies_annotations), comparison_dataframe

    def interface(self):
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    image = gr.Image()
                    start = gr.Button("Predict")
                # with gr.Column():
                #     comparison = gr.DataFrame()

            with gr.Row():
                output = gr.AnnotatedImage()

            demo.load(self.load_model)

            start.click(self.predict, inputs=[image], outputs=[output])
            # start.click(self.predict, inputs=[image], outputs=[output, comparison])

        self.demo = demo


def run():
    gui = Interface()
    gui.demo.launch()


if __name__ == "__main__":
    run()
