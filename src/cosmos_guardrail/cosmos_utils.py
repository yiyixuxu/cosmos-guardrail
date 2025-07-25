import os
import re

import numpy as np
import torch

from .utils import get_logger

import cv2

from retinaface.utils.nms.py_cpu_nms import py_cpu_nms


logger = get_logger(__name__)  # pylint: disable=invalid-name


def read_keyword_list_from_dir(folder_path: str) -> list[str]:
    """Read keyword list from all files in a folder."""
    output_list = []
    file_list = []
    # Get list of files in the folder
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            file_list.append(file)

    # Process each file
    for file in file_list:
        file_path = os.path.join(folder_path, file)
        try:
            with open(file_path, "r") as f:
                output_list.extend([line.strip() for line in f.readlines()])
        except Exception as e:
            logger.error(f"Error reading file {file}: {str(e)}")

    return output_list


def to_ascii(prompt: str) -> str:
    """Convert prompt to ASCII."""
    return re.sub(r"[^\x00-\x7F]+", " ", prompt)


def pixelate_face(face_img: np.ndarray, blocks: int = 5) -> np.ndarray:
    """
    Pixelate a face region by reducing resolution and then upscaling.

    Args:
        face_img: Face region to pixelate
        blocks: Number of blocks to divide the face into (in each dimension)

    Returns:
        Pixelated face region
    """
    h, w = face_img.shape[:2]
    # Shrink the image and scale back up to create pixelation effect
    temp = cv2.resize(face_img, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    return pixelated


# Adapted from https://github.com/biubug6/Pytorch_Retinaface/blob/master/detect.py
def filter_detected_boxes(boxes, scores, confidence_threshold, nms_threshold, top_k, keep_top_k):
    """Filter boxes based on confidence score and remove overlapping boxes using NMS."""
    # Keep detections with confidence above threshold
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # Sort by confidence and keep top K detections
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    scores = scores[order]

    # Run non-maximum-suppression (NMS) to remove overlapping boxes
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    dets = dets[:keep_top_k, :]
    boxes = dets[:, :-1]
    return boxes


# Adapted from https://github.com/biubug6/Pytorch_Retinaface/blob/master/utils/box_utils.py to handle batched inputs
def decode_batch(loc, priors, variances):
    """Decode batched locations from predictions using priors and variances.

    Args:
        loc (tensor): Batched location predictions for loc layers.
            Shape: [batch_size, num_priors, 4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors, 4]
        variances: (list[float]): Variances of prior boxes.

    Return:
        Decoded batched bounding box predictions
            Shape: [batch_size, num_priors, 4]
    """
    batch_size = loc.size(0)
    priors = priors.unsqueeze(0).expand(batch_size, -1, -1)

    boxes = torch.cat(
        (
            priors[:, :, :2] + loc[:, :, :2] * variances[0] * priors[:, :, 2:],
            priors[:, :, 2:] * torch.exp(loc[:, :, 2:] * variances[1]),
        ),
        dim=2,
    )

    boxes[:, :, :2] -= boxes[:, :, 2:] / 2
    boxes[:, :, 2:] += boxes[:, :, :2]
    return boxes


# Adapted from https://github.com/biubug6/Pytorch_Retinaface/blob/master/detect.py
def _check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    logger.debug("Missing keys:{}".format(len(missing_keys)))
    logger.debug("Unused checkpoint keys:{}".format(len(unused_pretrained_keys)))
    logger.debug("Used keys:{}".format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, "load NONE from pretrained checkpoint"
    return True


# Adapted from https://github.com/biubug6/Pytorch_Retinaface/blob/master/detect.py
def _remove_prefix(state_dict, prefix):
    """Old version of the model is stored with all names of parameters sharing common prefix 'module.'"""
    logger.debug("Removing prefix '{}'".format(prefix))

    def f(x):
        return x.split(prefix, 1)[-1] if x.startswith(prefix) else x

    return {f(key): value for key, value in state_dict.items()}


# Adapted from https://github.com/biubug6/Pytorch_Retinaface/blob/master/detect.py
def load_model(model, pretrained_path):
    logger.debug("Loading pretrained model from {}".format(pretrained_path))
    pretrained_dict = torch.load(pretrained_path, weights_only=True)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = _remove_prefix(pretrained_dict["state_dict"], "module.")
    else:
        pretrained_dict = _remove_prefix(pretrained_dict, "module.")
    _check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


UNSAFE_CATEGORIES = {
    "S1": "Violent Crimes.",
    "S2": "Non-Violent Crimes.",
    "S3": "Sex Crimes.",
    "S4": "Child Exploitation.",
    "S5": "Defamation.",
    "S6": "Specialized Advice.",
    "S7": "Privacy.",
    "S8": "Intellectual Property.",
    "S9": "Indiscriminate Weapons.",
    "S10": "Hate.",
    "S11": "Self-Harm.",
    "S12": "Sexual Content.",
    "S13": "Elections.",
    "s14": "Code Interpreter Abuse.",
}


CLASS_IDX_TO_NAME = {
    0: "Safe",
    1: "Sexual_Content",
    2: "Violence",
    3: "Drugs",
    4: "Child_Abuse",
    5: "Hate_and_Harassment",
    6: "Self-Harm",
}

# RetinaFace model constants from https://github.com/biubug6/Pytorch_Retinaface/blob/master/detect.py
TOP_K = 5_000
KEEP_TOP_K = 750
NMS_THRESHOLD = 0.4
