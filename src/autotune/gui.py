import gradio as gr
from autotune.tuner import Tuner

tuner = Tuner()

with gr.Blocks() as demo:
    with gr.Row():
        new_alias = gr.Textbox(label="Alias")
        class_id = gr.Dropdown(
            label="Class ID", choices=tuner.dataset.classnames, type="index", value=2
        )
        box_threshold = gr.Slider(
            minimum=0, maximum=1.0, step=0.01, value=0.2, interactive=True
        )
        text_threshold = gr.Slider(
            minimum=0, maximum=1.0, step=0.01, value=0.2, interactive=True
        )
        start = gr.Button(value="Start", variant="primary")
        stop = gr.Button(value="Stop", variant="stop")
    with gr.Row():
        progress = gr.Label(show_label=False)
        precision = gr.Label(
            show_label=False,
            value={"precision (allucinations)": 0, "recall (unrecognized)": 0},
        )
        preview = gr.AnnotatedImage(show_label=False)
    with gr.Row():
        history = gr.HighlightedText(
            show_label=False,
            color_map=tuner.color_map,
            value=tuner.display_history(tuner.llm.history),
        )
        progress.change(tuner.update_history, inputs=[progress], outputs=[history])

    start.click(
        tuner.pipeline,
        inputs=[new_alias, class_id],
        outputs=[new_alias, progress, precision, preview],
    )
    stop.click()


def run():
    demo.launch(allowed_paths=[DATASET_PATH], share=False, show_error=True)


if __name__ == "__main__":
    run()
