import gradio as gr
from DINO import Dino
from metrics import confusion_matrix
import os
from dataset import Dataset
from tqdm import tqdm
from autotune.generate import AliasGenerator
import atexit
from dotenv import load_dotenv


class Tuner:
    def __init__(self):
        self.color_map: dict[str, str] = {
            "H PR": "green",
            "H RC": "green",
            "L PR": "yellow",
            "L RC": "yellow",
            "B PR": "red",
            "B RC": "red",
            "T PR": "blue",
            "T RC": "blue",
        }
        self.alias = ""
        self.stop = False

    def load(self):
        load_dotenv()
        cwd = os.getcwd()
        DATASET_PATH = os.path.join(cwd, "dataset_sample")
        try:
            DATASET_PATH = os.getenv("DATASET_PATH") or DATASET_PATH
            if not DATASET_PATH:
                raise ValueError("DATASET_PATH is not set")
            self.dataset = Dataset(DATASET_PATH)
            self.dataset.populate()
        except Exception as e:
            raise e

        try:
            self.model = Dino(size="small")
            print("DINO Model loaded successfully")
        except Exception as e:
            raise e

        try:
            self.llm = AliasGenerator()
            print("AliasGenerator loaded successfully")
        except Exception as e:
            raise e

        # on quit save the history
        atexit.register(self.llm.save, self.llm.path)

    def benchmark(
        self,
        alias,
        class_id,
        box_threshold=0.1,
        text_threshold=0.1,
    ):
        total = len(self.dataset.images)
        metrics = [0, 0, 0]

        annotatedimage = None
        for i, img in enumerate(self.dataset.images):
            img_bgr = img.data
            result = self.model.predict(img_bgr, alias, box_threshold, text_threshold)

            metric = confusion_matrix(
                predictions=result,
                ground_truths=img.grounds,
                device=self.model.device,
                class_id=class_id,
            )

            metrics[0] += metric[0]
            metrics[1] += metric[1]
            metrics[2] += metric[2]

            # every 10 iterations, update the gradio annotatedImage
            if i % 10 == 0:
                annotatedimage = self.display_preview(
                    result, img_bgr, self.dataset.classnames[class_id]
                )

            precision = (
                metrics[0] / (metrics[0] + metrics[1])
                if metrics[0] + metrics[1] > 0
                else 0.0
            )  # Or raise an exception
            recall = (
                metrics[0] / (metrics[0] + metrics[2])
                if metrics[0] + metrics[2] > 0
                else 0.0
            )

            # Yield progress and metrics on every iteration
            yield (
                {f"progress {i}/{total}": i / total},
                {
                    "precision": precision,
                    "recall": recall,
                },
                annotatedimage,
            )

    def pipeline(self, new_alias, class_id=0):
        """Recursive pipeline for finding the best aliases"""

        # Iterate over the benchmark generator
        for status, progress_metrics, annotatedimage in self.benchmark(
            new_alias, class_id
        ):
            self.llm.update_history(
                new_alias, (progress_metrics["precision"], progress_metrics["recall"])
            )

            # Yield the current progress and metrics to update the Gradio interface
            yield status, progress_metrics, annotatedimage

        # Extract the final metrics and annotation from the last yielded value
        final_metrics = (progress_metrics["precision"], progress_metrics["recall"])

        # Recursively call pipeline with improved alias based on final metrics
        if (
            self.llm.should_continue(final_metrics) and not self.stop
        ):  # Add a stopping condition
            new_alias = self.llm.improve(final_metrics, class_id)
            self.alias = new_alias
            yield from self.pipeline(new_alias, class_id)

    def update_alias(self):
        return self.alias

    def update_history(self, progress: dict):
        for status in progress.values():
            if status == 1.0:
                return self.display_history(self.llm.history)

    def display_history(self, history) -> list[tuple[str, str]]:
        """
        Args:
            history: dict[str, tuple[float, float]]
        Returns:
            list[tuple[str, str]]= list of (word, category) tuples
        """
        best_precision = 0.3
        best_pr_alias = ""
        best_recall = 0.3
        best_rc_alias = ""

        good_pr_aliases = []
        good_rc_aliases = []

        bad_precision = 0.2
        bad_pr_aliases = []
        bad_recall = 0.2
        bad_rc_aliases = []

        worst_precision = 0.1
        worst_pr_aliases = []
        worst_recall = 0.1
        worst_rc_aliases = []

        for alias, (precision, recall) in history.items():
            if precision < worst_precision:
                worst_pr_aliases.append(alias)
            if recall < worst_recall:
                worst_rc_aliases.append(alias)
            elif precision < bad_precision:
                bad_pr_aliases.append(alias)
            elif recall < bad_recall:
                bad_rc_aliases.append(alias)
            elif precision > best_precision:
                best_precision = precision
                good_pr_aliases.append(best_pr_alias)
                best_pr_alias = alias
            elif recall > best_recall:
                best_recall = recall
                good_rc_aliases.append(best_rc_alias)
                best_rc_alias = alias
            elif precision > bad_precision > best_precision:
                good_pr_aliases.append(alias)
            elif recall > bad_recall > best_recall:
                good_rc_aliases.append(alias)

        performances: list[tuple[str, str]] = []
        for alias in good_pr_aliases:
            performances.append((alias, "H PR"))
        for alias in good_rc_aliases:
            performances.append((alias, "H RC"))
        for alias in bad_pr_aliases:
            performances.append((alias, "L PR"))
        for alias in bad_rc_aliases:
            performances.append((alias, "L RC"))
        for alias in worst_pr_aliases:
            performances.append((alias, "B PR"))
        for alias in worst_rc_aliases:
            performances.append((alias, "B RC"))
        performances.append((best_pr_alias, "T PR"))
        performances.append((best_rc_alias, "T RC"))

        return performances

    def display_preview(self, boxes, image, classname):
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

    def load_interface(self):
        with gr.Blocks() as interface:
            with gr.Row():
                # Load inputs
                new_alias = gr.Textbox(label="Alias")
                class_id = gr.Dropdown(
                    label="Class ID", choices=self.dataset.classnames, type="index"
                )

                box_threshold = gr.Slider(
                    minimum=0, maximum=1.0, step=0.01, value=0.2, interactive=True
                )

                text_threshold = gr.Slider(
                    minimum=0, maximum=1.0, step=0.01, value=0.2, interactive=True
                )

                start = gr.Button(value="Start", variant="primary")
                stop = gr.Button("Stop Pipeline", variant="stop")
            with gr.Row():
                # Load progress outputs
                progress = gr.Label(show_label=False)
                performance = gr.Label(
                    show_label=False,
                    value={"precision (allucinations)": 0, "recall (unrecognized)": 0},
                )
                preview = gr.AnnotatedImage(show_label=False)
            with gr.Row():
                # Load history output
                # FIX: disappears
                history = gr.HighlightedText(
                    show_label=False,
                    color_map=self.color_map,
                    value=self.display_history(self.llm.history),
                )

            start.click(
                self.pipeline,
                inputs=[new_alias, class_id],
                outputs=[progress, performance, preview],
                trigger_mode="once",
            )

            # Stop button
            stop.click(self.stop_pipeline, None, None, queue=False)
            progress.change(self.update_history, inputs=[progress], outputs=[history])
            progress.change(self.update_alias, outputs=[new_alias])

        self.interface = interface

    def stop_pipeline(self):
        self.stop = True


def run():
    tuner = Tuner()
    tuner.load()
    tuner.load_interface()
    tuner.interface.launch()


if __name__ == "__main__":
    run()
