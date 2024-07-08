import torch
import torchvision.ops as ops
import random
import cv2 as cv
from typing import Tuple


def draw_boxes(boxes, scores, image):
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv.putText(
            image, f"{score:.2f}", (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )


def show(image):
    try:
        cv.imshow("results", image)
    except KeyboardInterrupt:
        exit(0)

    key = cv.waitKey(0) & 0xFF
    if key == 27 or key == ord("q") or key == 3:
        exit(0)


def remove_outliers(boxes: torch.Tensor, shape: Tuple[int, int], threshold=0.4):
    """Remove boxes that are outliers in terms of size
    Args:
        boxes: list of boxes
        shape: shape of the image (height, width)
        threshold: threshold for removing outliers
    Returns:
        list of boxes after removing outliers
    """

    # remove boxes with width or height more than 0.4 of the image size
    box_width = boxes[:, 2] - boxes[:, 0]
    box_height = boxes[:, 3] - boxes[:, 1]
    box_width = box_width / shape[1]
    box_height = box_height / shape[0]

    bigger_indices = (box_width > threshold) | (box_height > threshold)
    return ~bigger_indices


def run():
    image = cv.imread(
        "/Users/francescotacinelli/Developer/automata/dataset_sample/0d0d3b66-4200077108_1914_159.jpg"
    )

    boxes1 = torch.tensor(
        [
            [132.7131, 253.2612, 512.2308, 997.3495],
            [1303.8726, 257.2816, 1913.9471, 696.1411],
            [2444.0117, 190.2341, 2574.9182, 1642.1160],
            [480.0776, 293.0996, 714.9066, 914.5599],
            [2262.3838, 202.6952, 2381.3877, 1628.4315],
            [2083.4785, 214.4127, 2198.0369, 1621.1300],
            [1299.4614, 675.2523, 1900.9467, 1095.2924],
            [15.9392, 150.9510, 2579.6213, 1629.4576],
            [1762.6246, 1426.5515, 1859.5858, 1603.0471],
            [1292.0011, 670.8241, 1800.9141, 1020.0206],
        ],
        device="mps:0",
    )
    scores1 = torch.tensor(
        [0.3180, 0.2791, 0.2358, 0.2340, 0.2234, 0.1756, 0.1725, 0.1638, 0.1633],
        device="mps:0",
    )
    boxes2 = torch.tensor(
        [
            [820.0154, 116.4760, 1291.2638, 660.6686],
            [1287.9030, 1081.2302, 1796.9094, 1576.5278],
            [134.8970, 981.3054, 678.3124, 1505.2396],
            [851.4787, 1014.0995, 1300.9182, 1531.4662],
            [10.7306, 68.6023, 2582.2253, 1657.9371],
            [131.4253, 252.3970, 512.6182, 996.0939],
            [799.1781, 659.3257, 1346.1393, 1021.3464],
            [1304.0260, 258.9301, 1915.3949, 693.2707],
            [479.3557, 307.5020, 714.8411, 906.8140],
            [548.3494, 692.5928, 863.2629, 1171.5288],
            [1297.0011, 676.8241, 1900.9141, 1090.0206],
        ],
        device="mps:0",
    )
    scores2 = torch.tensor(
        [
            0.5121,
            0.4675,
            0.4178,
            0.4057,
            0.3653,
            0.3566,
            0.3509,
            0.3221,
            0.2925,
            0.2632,
            0.2223,
        ],
        device="mps:0",
    )

    # Calculate IoU
    ious = ops.box_iou(boxes1, boxes2)

    print("ious = ", ious)

    threshold = 0.5
    overlapping = ious >= threshold

    # print number of overlapping pairs
    print("overlapping = ", overlapping.sum())

    # Normalize the scores
    scores1_norm = scores1 / scores1.max()
    scores2_norm = scores2 / scores2.max()

    # for each set of boxes, get the scores of overlapping ious
    overscores1 = scores1[overlapping[0]]
    overscores2 = scores2[overlapping[1]]

    # Make a boolean mask for the boxes with lower scores in each pair
    keep_scores1 = overscores1 < overscores2
    keep_scores2 = overscores2 < overscores1

    # remove overlapping boxes from the set with lower score

    draw_boxes(boxes1, scores1, image)
    draw_boxes(boxes2, scores2, image)

    # remove overlapping boxes from the set with lower score

    show(image)


# # Normalize the scores
# scores1_norm = scores1 / scores1.max()
# scores2_norm = scores2 / scores2.max()
#
# # Make a boolean mask for the boxes with lower scores in each pair
# low_scores_1 = scores1_norm > scores2_norm
#
# over1 = overlapping[0]
# over2 = overlapping[1]

if __name__ == "__main__":
    run()
