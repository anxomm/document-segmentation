import matplotlib.pyplot as plt
import numpy as np
from skimage import color, io

import os


def readImage(path):

    try:
        int(path)
        filename = os.path.join(os.getcwd(), f"Examples/doc{path}.jpg")
    except ValueError:
        filename = os.path.join(os.getcwd(), path)

    rgb = io.imread(filename)
    gray = color.rgb2gray(rgb)
    gray = gray / 255 if gray.dtype == np.uint8 else gray

    return rgb, gray


def plot(images, points=[], titles=["Original", "Output"], gray=True):

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(min(4*n, 12), 4))

    axes[0].imshow(images[0])  # original image
    axes[0].set_title(titles[0])

    for i in range(1, n):
        image = images[i]
        title = titles[i] if i < len(titles) else f"Output{i+1}"
        if (gray):
            axes[i].imshow(image, cmap=plt.cm.gray, vmin=0, vmax=1)
        else:
            axes[i].imshow(image)
        axes[i].set_title(title)

    if (len(points) > 0):
        for ax, data in points:
            for (x, y) in data:
                axes[ax].plot(y, x, "ro")

    fig.tight_layout()
    plt.show()


def save_images(images, names, path, relative=True):

    assert len(images) == len(names), "Must include one name per image"

    if (relative):
        path = os.path.join(os.getcwd(), path)

    for i in range(len(images)):
        image = (images[i] * 255).astype(np.uint8)
        name = os.path.join(path, f"{names[i]}.jpg")
        io.imsave(name, image)
