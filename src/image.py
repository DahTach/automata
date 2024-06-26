import pathlib
from typing import List, Tuple

# import jpeg4py as jpeg
import cv2 as cv

sample = "/Users/francescotacinelli/Developer/datasets/pallets_sorted/labeled/images/0d0d3b66-4200077108_1914_159.jpg"


class Image:
    def __init__(self, path):
        self.path = pathlib.Path(path)

    @property
    def data(self):
        # if self.path.suffix in [".jpg", ".jpeg"]:
        #     return jpeg.JPEG("test.jpg").decode()
        return cv.imread(str(self.path.resolve()))

    @property
    def framing(self):
        parent = self.path.parent
        return parent.name

    @property
    def grounds(self) -> List[Tuple[int, List[float]]]:
        path = self.path.with_suffix(".txt")

        # if path doesn't exist throw error
        if not path.exists():
            raise FileNotFoundError(f"Ground truth file for {self.path} not found")

        with open(path, "r") as f:
            lines = f.readlines()
            grounds = []
            for line in lines:
                class_id, *bbx = line.split()
                grounds.append((int(class_id), [float(coord) for coord in bbx]))
            return grounds


def test(image):
    img = Image(image)
    print(img.grounds)


if __name__ == "__main__":
    test(sample)
