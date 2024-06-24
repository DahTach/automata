import os
import progressbar
import warnings
import cv2 as cv
import numpy as np
import pathlib
import torch


def get_device():
    if torch.cuda.is_available():
        print("using cuda")
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("using mps")
        return "mps"
    else:
        print("using cpu")
        return "cpu"


def get_captions(captions):
    """Inverts a dictionary with list values.
    Args:
        d: The dictionary to invert.
    Returns:
        The inverted dictionary.
    """

    inverted_dict = {}
    for key, value in captions.items():
        for item in value:
            inverted_dict.setdefault(item, key)  # Set default value for repeated values
    return inverted_dict


pbar = None


def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None


def get_image_files(path):
    images = []
    # recursively search for images in the folder
    for folder, _, files in os.walk(path):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                images.append(os.path.join(folder, file))

    return images


def show(img: np.ndarray):
    try:
        cv.imshow("results", img)
    except KeyboardInterrupt:
        exit(0)
    key = cv.waitKey(0) & 0xFF
    if key == 27 or key == ord("q") or key == 3:
        exit(0)


def filter_warnings():
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", FutureWarning)
    warnings.filterwarnings(
        "ignore",
        message="The `device` argument is deprecated and will be removed in v5 of Transformers.",
    )
    warnings.filterwarnings("ignore", message="torch.utils.checkpoint")
    warnings.filterwarnings(
        "ignore", message="None of the inputs have requires_grad=True"
    )
    warnings.filterwarnings("ignore", message="To copy construct from a tensor")

    warnings.filterwarnings("ignore", message="torch.meshgrid")
    warnings.filterwarnings("ignore", message="torch.broadcast_tensors")
    # warnings.filterwarnings("ignore", message="Warning: CUDA not available. GroundingDINO will run very slowly.")


def save_image(image, path_name):
    # get pathlib path
    path = pathlib.Path(path_name)
    result_dir = path.parent.parent / f"{path.parent.name}_results"
    result_path = result_dir / path.name
    cv.imwrite(str(result_path.resolve()), image)


def filter_phi_warnings():
    warnings.filterwarnings(
        "ignore",
        message="`do_sample` is set to `False`",
    )
    warnings.filterwarnings(
        "ignore",
        message="You are not running the flash-attention implementation",
    )
