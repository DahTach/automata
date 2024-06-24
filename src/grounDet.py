from metrics import confusion_matrix
from image import Image
from typing import Tuple
from DINO import Dino


class Detector:
    def __init__(self, model):
        self.detector = model

    def test_alias(
        self, image: Image, alias, class_id=2, box_threshold=0.5, text_threshold=0.5
    ) -> Tuple[int, int, int]:
        boxes, confidences = self.detector.predict(
            image=image.data,
            prompt=alias,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        ground_truths = image.grounds

        return confusion_matrix(boxes, ground_truths, class_id)


def test_detector():
    model = Dino()
    detector = Detector(model)
    image = Image("test.jpg")
    metrics = detector.test_alias(image, "box")
    print(metrics)


if __name__ == "__main__":
    test_detector()
