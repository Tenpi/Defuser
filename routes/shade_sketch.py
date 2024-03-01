import tensorflow.compat.v1 as tf
from .functions import pil_to_cv2
import numpy as np
import cv2
import os

dirname = os.path.dirname(__file__)

shader_model = None
norm_model = None
smoother_model = None

def cond_to_pos(cond):
    cond_pos_rel = {
        "002": [0, 0, -1],
        "110": [0, 1, -1], "210": [1, 1, -1], "310": [1, 0, -1], "410": [1, -1, -1], "510": [0, -1, -1],
        "610": [-1, -1, -1], "710": [-1, 0, -1], "810": [-1, 1, -1],
        "120": [0, 1, 0], "220": [1, 1, 0], "320": [1, 0, 0], "420": [1, -1, 0], "520": [0, -1, 0], "620": [-1, -1, 0],
        "720": [-1, 0, 0], "820": [-1, 1, 0],
        "130": [0, 1, 1], "230": [1, 1, 1], "330": [1, 0, 1], "430": [1, -1, 1], "530": [0, -1, 1], "630": [-1, -1, 1],
        "730": [-1, 0, 1], "830": [-1, 1, 1],
        "001": [0, 0, 1]
    }
    return cond_pos_rel[cond]

def normalize_cond(cond_str):
    _cond_str = cond_str.strip()

    if len(_cond_str) == 3:
        return cond_to_pos(_cond_str)

    if "," in _cond_str:
        raw_cond = _cond_str.replace("[", "").replace("]", "").split(",")
        if len(raw_cond) == 3:
            return raw_cond

    return [-1, 1, -1]


def shade_sketch(input, output, direction="810", size=320, threshold=200, use_smooth=True, use_norm=True):
    global shader_model
    global smoother_model
    global norm_model

    if not smoother_model:
        with tf.gfile.GFile(os.path.join(dirname, "../models/misc/linesmoother.pb"), "rb") as f:
            smoother_model = tf.GraphDef()
            smoother_model.ParseFromString(f.read())
    tf.import_graph_def(smoother_model)

    if not norm_model:
        with tf.gfile.GFile(os.path.join(dirname, "../models/misc/linenorm.pb"), "rb") as f:
            norm_model = tf.GraphDef()
            norm_model.ParseFromString(f.read())
    tf.import_graph_def(norm_model)

    if not shader_model:
        with tf.gfile.GFile(os.path.join(dirname, "../models/misc/lineshader.pb"), "rb") as f:
            shader_model = tf.GraphDef()
            shader_model.ParseFromString(f.read())
    tf.import_graph_def(shader_model)

    with tf.Session() as sess:
        img = pil_to_cv2(input)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]

        imgrs = cv2.resize(img, (size, size))

        if threshold > 0:
            _, imgrs = cv2.threshold(imgrs, threshold, 255, cv2.THRESH_BINARY)

        tensors = np.reshape(imgrs, (1, size, size, 1)).astype(np.float32) / 255.
        ctensors = np.expand_dims(normalize_cond(direction), 0)

        if use_smooth or threshold > 0:
            tensors = sess.run("conv2d_9/Sigmoid:0", feed_dict={"input_1:0": tensors})
            smoothResult = tensors

        if use_norm:
            tensors = sess.run("conv2d_9/Sigmoid_1:0", feed_dict={"input_1_1:0": tensors})
            normResult = tensors

        tensors = sess.run("conv2d_139/Tanh:0", feed_dict={"input_1_2:0": ctensors, "input_2:0": 1. - tensors})
        shadeResult = tensors

        shade = (1 - (np.squeeze(shadeResult) + 1) / 2) * 255.
        shade = cv2.resize(shade, (w, h))

        comp = 0.8 * img + 0.2 * shade

        cv2.imwrite(output, comp)