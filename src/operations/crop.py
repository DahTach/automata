import cv2 as cv
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, help="Path to the image")


def show(img: np.ndarray):
    try:
        cv.imshow("results", img)
    except KeyboardInterrupt:
        exit(0)
    key = cv.waitKey(0) & 0xFF
    if key == 27 or key == ord("q") or key == 3:
        exit(0)


# define the points of the rectangle
x1 = 0
x2 = 2060
y1 = 330
y2 = 1860

box = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]


def main():
    args = parser.parse_args()

    image = cv.imread(args.image)

    # define the function to draw the rectangle
    cv.rectangle(image, box[0], box[2], (0, 255, 0), 2)
    show(image)


if __name__ == "__main__":
    main()
