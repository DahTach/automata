import torch
import torchvision.ops as ops
from typing import List, Tuple
import matplotlib.pyplot as plt


class Metrics:
    def __init__(
        self,
        confusion_matrices: List[List[Tuple[int, int, int]]],
        confidences: List[float],
    ):
        self.true_positives = [[0] * len(confidences)]
        self.false_positives = [[0] * len(confidences)]
        self.false_negatives = [[0] * len(confidences)]
        self.confidences = confidences
        self.process_matrices(confusion_matrices)

    def process_matrices(self, matrices):
        for img in matrices:
            for conf, matrix in enumerate(img):
                tp, fp, fn = matrix
                self.true_positives[conf] += tp
                self.false_positives[conf] += fp
                self.false_negatives[conf] += fn

    def precision(self, conf):
        return self.true_positives[conf] / (
            self.true_positives[conf] + self.false_positives[conf]
        )

    def recall(self, conf):
        return self.true_positives[conf] / (
            self.true_positives[conf] + self.false_negatives[conf]
        )

    def f1(self, conf):
        precision = self.precision(conf)
        recall = self.recall(conf)
        return 2 * (precision * recall) / (precision + recall)

    def precision_recall_curve(self):
        return [
            (self.precision(conf), self.recall(conf))
            for conf in range(len(self.confidences))
        ]

    def plot_precision_recall_curve(self):
        curve = self.precision_recall_curve()
        plt.plot(*zip(*curve))
        plt.show()


def confusion_matrix(
    predictions: torch.Tensor,
    ground_truths: List[Tuple[int, List[float]]],
    device,
    class_id,
    iou_threshold=0.5,
) -> List[int]:
    class_prs = predictions
    class_grs = [bbox for idx, bbox in ground_truths if idx == class_id]

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    if class_grs and class_prs.numel() != 0:
        gr_bbxs = ops.box_convert(
            boxes=torch.tensor(class_grs),
            in_fmt="cxcywh",
            out_fmt="xyxy",
        ).to(device)

        pr_bbxs = class_prs
        iou_matrix = ops.box_iou(gr_bbxs, pr_bbxs)
        matched_indices = torch.where(iou_matrix >= iou_threshold)

        true_positives = matched_indices[0].unique().numel()

        false_positives = len(class_prs) - true_positives

        false_negatives = len(class_grs) - true_positives
    elif class_prs.numel() == 0:
        false_negatives = 0
    elif not class_grs:
        false_positives = 0

    return [true_positives, false_positives, false_negatives]
