import ioutilities as io
from process import *
import sys

if __name__ == '__main__':

    if (len(sys.argv) < 2):
        print("Usage:    python main.py <path_to_image>  # Read an image from your computer")
        print("          python main.py <number>         # Read doc<number>.jpg image from 'Examples'")
        exit(-1)

    # Read input
    original, grayscale = io.readImage(sys.argv[1])

    # Preprocess
    print("Preprocessing image...")
    preprocessed = preprocess(grayscale)

    # Detect paper
    print("Searching for corners...")
    edges, corners = detectCorners(*preprocessed)

    # Change perspective
    print("Changing perspective...")
    warped = changePerspective(grayscale, corners)
    warpedrgb = changePerspective(original, corners).astype(np.uint8)

    # Split foreground (text, pen marks) from background (white, coffee marks)
    print("Detecting text...")
    thrs, color_mask = detectText(warpedrgb, warped, block_size=101, offset=0.06, black_std=7)

    # Remove pen marks
    print("Removing pen marks...")
    pen_mask = removeColor(warpedrgb, color_mask, radius=30, black_std=7)
    pen_mask2 = hysteresis(pen_mask, 1 - thrs)
    out = 1 - ((1 - thrs) - pen_mask2)

    # Show result
    # io.plot([original, out], [(0, corners[0:4])])
    io.save_images([out], ["output"], ".")
