"""
Character Detection

The goal of this task is to experiment with template matching techniques. Specifically, the task is to find ALL of
the coordinates where a specific character appears using template matching.

There are 3 sub tasks:
1. Detect character 'a'.
2. Detect character 'b'.
3. Detect character 'c'.

You need to customize your own templates. The templates containing character 'a', 'b' and 'c' should be named as
'a.jpg', 'b.jpg', 'c.jpg' and stored in './data/' folder.

The functions you implement in EdgeDetection.py are of great help.

Do NOT modify the code provided.
Do NOT use any API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os

import numpy as np
import utils
from EdgeDetection import *   # you could modify this line


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img_path", type=str, default="./data/characters.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--template_path", type=str, default="",
        choices=["./data/a.pgm", "./data/b.pgm", "./data/c.pgm"],
        help="path to the template image")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./results/",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args


def correlation_coefficient(patch_edges, template_edges):
    patch_mean = np.sum(patch_edges) / patch_edges.size
    template_mean = np.sum(template_edges) / template_edges.size

    patch_var = (patch_edges - patch_mean)
    template_var = (template_edges - template_mean)

    correlation = np.sum(utils.elementwise_mul(patch_var, template_var)) / np.sqrt(
        np.sum(np.square(patch_var)) * np.sum(np.square(template_var)))

    return correlation


def detect(img, template):
    """Detect a given character, i.e., the character in the template image.

    Args:
        img: nested list (int), image that contains character to be detected.
        template: nested list (int), template image.

    Returns:
        coordinates: list (tuple), a list whose elements are coordinates where the character appears.
            format of the tuple: (x (int), y (int)), x and y are integers.
            x: row that the character appears (starts from 0).
            y: column that the character appears (starts from 0).
    """
    # TODO: implement this function.
    # raise NotImplementedError

    # template = cv2.resize(template, (0,0), fx=0.5, fy=0.5)
    # template_H = template.shape[0]
    # template_W = template.shape[1]

    template_H = len(template)
    template_W = len(template[0])

    #kernel_x = task1.sobel_x
    #kernel_y = task1.sobel_y
    # threshold = 0.775
    threshold = 0.94
    # print(template.shape[:])

    # template_edge_x = convolve2d(template,kernel_x)
    # template_edge_y = convolve2d(template,kernel_y)
    # template_edges = edge_magnitude(template_edge_x,template_edge_y)
    # cv2.imwrite('template_edges.jpg',template_edges)

    template = normalize(np.asarray(template))

    coordinates = [[]]
    for i in range(len(img) - template_H):
        for j in range(len(img[0]) - template_W):
            """if i+template_H>len(img) or j+template_W>len(img):
                continue"""

            img_patch = utils.crop(img, i, i + template_H, j, j + template_W)
            # patch_edge_x = convolve2d(img_patch,kernel_x)
            # patch_edge_y = convolve2d(img_patch,kernel_y)
            # patch_edges = edge_magnitude(patch_edge_x,patch_edge_y)
            img_patch = normalize(np.asarray(img_patch))
            correlation = correlation_coefficient(img_patch, template)

            if correlation > threshold:
                coordinates.append((i, j))

    del coordinates[0]

    return coordinates


def save_results(coordinates, template, template_name, rs_directory):
    results = {}
    results["coordinates"] = sorted(coordinates, key=lambda x: x[0])
    results["templat_size"] = (len(template), len(template[0]))
    with open(os.path.join(rs_directory, template_name), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()

    img = read_image(args.img_path)
    template = read_image(args.template_path)

    coordinates = detect(img, template)

    template_name = "{}.json".format(os.path.splitext(os.path.split(args.template_path)[1])[0])
    save_results(coordinates, template, template_name, args.rs_directory)


if __name__ == "__main__":
    main()
