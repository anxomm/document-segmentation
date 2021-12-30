import numpy as np
from scipy import ndimage
from skimage import exposure, feature, filters, transform, morphology


def preprocess(gray):

    hist, _ = exposure.cumulative_distribution(gray, nbins=256)

    if (hist[128] < 0.1):  # very light
        corrected = exposure.adjust_gamma(gray, gamma=8.5, gain=2)
        median_size, high_threshold = 7, 0.1
    elif (hist[128] < 0.3):  # light image
        corrected = exposure.adjust_gamma(gray, gamma=2, gain=1.2)
        median_size, high_threshold = 10, 0.1
    else:
        corrected = gray
        median_size, high_threshold = 10, 0.2

    return corrected, median_size, high_threshold


def detectCorners(gray, median_size=10, high_threshold=0.2):

    # Decrease resolution
    resize_x, resize_y = 300, 250
    resize = transform.resize(gray, (resize_x, resize_y), anti_aliasing=True)

    # Clear text and calculate edges/corners
    median = ndimage.median_filter(resize, median_size)
    edges = feature.canny(median, 1, low_threshold=0.01, high_threshold=high_threshold).astype(gray.dtype)
    corners, (h, theta, d) = strongestCorners(edges)
    # plot_hough(edges, h, theta, d)

    # Extrapolate the position of the corners to the original resolution
    m, n = gray.shape
    corners = np.array([[x * (m/resize_x), y * (n/resize_y)] for (x, y) in corners])

    return edges, corners.astype(np.int32)


def strongestCorners(edges):

    # Find the strongest lines
    h, theta, d = transform.hough_line(edges)
    peaks = np.zeros((4, 5))
    col1 = edges.shape[1]

    idx = 0
    for _, angle, dist in zip(*transform.hough_line_peaks(h, theta, d, min_distance=110, threshold=30, num_peaks=4)):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - col1 * np.cos(angle)) / np.sin(angle)

        peaks[idx, :] = 0, y0, col1, y1, angle
        idx += 1

    # Convert radians to degrees, and transform to a positive angle between 0-180º
    peaks[:, 4] = np.abs(peaks[:, 4] * 180/np.pi)
    peaks[:, 4] = [angle if angle < 180 else 360 - angle for angle in peaks[:, 4]]

    # Calculate the intersection between the vertical and the horizontal lines
    mean_angle = np.mean(peaks[:, 4])
    hor = peaks[peaks[:, 4] < mean_angle]
    vert = peaks[peaks[:, 4] > mean_angle]

    def line_intersection(line1, line2):
        '''
        Code adapted from Paul Draper: https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
        '''
        xdiff = (line1[0] - line1[2], line2[0] - line2[2])
        ydiff = (line1[1] - line1[3], line2[1] - line2[3])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('lines do not intersect')

        d = (det(line1[0: 2], line1[2: 4]), det(line2[0: 2], line2[2: 4]))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y

    peaks = np.zeros((4, 2))
    peaks[0] = line_intersection(hor[0, :], vert[0, :])[:: -1]
    peaks[1] = line_intersection(hor[0, :], vert[1, :])[:: -1]
    peaks[2] = line_intersection(hor[1, :], vert[0, :])[:: -1]
    peaks[3] = line_intersection(hor[1, :], vert[1, :])[:: -1]

    # Sort corners as: top-left, top-right, bottom-left, bottom-right based on the x position
    peaks = peaks[np.argsort(peaks[:, 0])]
    if (peaks[0, 1] > peaks[1, 1]):
        peaks[[0, 1]] = peaks[[1, 0]]

    if (peaks[2, 1] > peaks[3, 1]):
        peaks[[2, 3]] = peaks[[3, 2]]

    return peaks, (h, theta, d)


def changePerspective(input, corners):

    # Correct the position of the corners to a bit inside the paper
    c1 = np.hypot(corners[2, 1] - corners[0, 1], corners[2, 0] - corners[0, 0]) / 70
    c2 = np.hypot(corners[1, 1] - corners[0, 1], corners[1, 0] - corners[0, 0]) / 50
    correction = np.array([[c1, c2], [c1, -c2], [-c1, c2], [-c1, -c2]])
    corners2 = corners + correction.astype(np.int32)

    # Adjust the perspective
    m = np.max([corners2[2, 0], corners2[3, 0]])
    n = np.max([corners2[1, 1], corners2[3, 1]])

    src = np.array([(0, 0), (n, 0), (0, m), (n, m)])
    dst = np.array([(y, x) for (x, y) in corners2])

    tform3 = transform.ProjectiveTransform()
    tform3.estimate(src, dst)

    return transform.warp(input, tform3, output_shape=(m, n), mode="constant", cval=0.5, preserve_range=True)


def iscolor(p, std=5):
    return np.std(p) >= std


def detectText(rgb, gray, block_size=201, offset=0.04, black_std=7):

    # Remove background
    local_thresh = filters.threshold_local(gray, block_size, offset=offset)
    background = gray > local_thresh

    # Detect black text
    def isblack(p): return np.std(p) < black_std

    text = (background == 0)
    result_black = np.apply_along_axis(isblack, 1, rgb[text])
    blacks = np.zeros_like(gray)
    blacks[text] = result_black

    # The rest is color (pen, black text tinted...)
    color = 1 - background - blacks

    return background.astype(np.float32), color


def removeColor(rgb, color, radius=8, black_std=10):

    def image_convolve_mask(image, list_points, radius):
        '''
        Code adapted from Scott at: https://stackoverflow.com/questions/29775700/image-convolution-at-specific-points
        '''
        rows, cols, _ = image.shape
        padded_image = np.pad(image, ((2*radius, 2*radius), (2*radius, 2*radius), (0, 0)), 'constant', constant_values=0)
        contrast = np.zeros((padded_image.shape[0], padded_image.shape[1], 3, 9))

        offset1, offset2 = radius, 3 * radius + 1
        ones = morphology.disk(radius) == 1

        def calculate(shift_x, shift_y, idx):
            i0, i1 = p[0] + offset1 + shift_y, p[0] + offset2 + shift_y
            j0, j1 = p[1] + offset1 + shift_x, p[1] + offset2 + shift_x
            i, j = p[0] + radius*2, p[1] + radius*2

            patch = padded_image[i0: i1, j0: j1, 0]
            contrast[i, j, 0, idx] = np.median(patch[ones])

            patch = padded_image[i0: i1, j0: j1, 1]
            contrast[i, j, 1, idx] = np.median(patch[ones])

            patch = padded_image[i0: i1, j0: j1, 2]
            contrast[i, j, 2, idx] = np.median(patch[ones])

        for p in list_points:
            calculate(0, 0, 0)              # center
            calculate(-radius, 0, 1)        # left
            calculate(radius, 0, 2)         # right
            calculate(0, -radius, 3)        # top
            calculate(0, radius, 4)         # down
            calculate(-radius, -radius, 5)  # top-left
            calculate(radius, -radius, 6)   # top-right
            calculate(-radius, radius, 7)   # bottom-left
            calculate(radius, radius, 8)    # bottom-left

        return contrast[2*radius: rows + 2*radius, 2*radius: cols + 2*radius, :, :]

    # Get the color of the background at each color point based on the median value of a window at 9 directions
    i, j = np.where(color == 1)
    points = list(zip(i, j))
    medianMask = image_convolve_mask(rgb, points, radius)
    mask = np.zeros_like(color)

    for p in points:
        i, j = p[0], p[1]
        c = medianMask[i, j, :, :]

        # Check if the point is tinted by marks such as coffee droplets
        isTinted = False
        for idx in range(0, 9):

            if (iscolor(c[:, idx])):  # the background is colored, then it is a tinted area
                isTinted = True

                r, g, b = medianMask[i, j, :, idx]  # color of the background in one direction
                r2, g2, b2 = rgb[i, j, :]  # color at a specific point

                # If the point is tinted of the same color as the background, then it is text (black/gray/white)
                s1, s2, s3 = abs(r - r2), abs(g - g2), abs(b - b2)
                if (iscolor([s1, s2, s3], black_std)):
                    mask[i, j] = 1

                break

        # The point is a pen mark
        if (not isTinted):
            mask[i, j] = 1

    return mask


def hysteresis(strong, weak):

    im = np.zeros(strong.shape, np.float32)
    im[weak == 1] = 0.5
    im[strong == 1] = 1

    return filters.apply_hysteresis_threshold(im, 0.1, 0.9)


def plot_hough(edges, h, theta, d):
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    plt.tight_layout()

    ax0.imshow(edges, cmap=plt.cm.gray)
    ax0.set_title('Input image')
    ax0.set_axis_off()
    ax1.imshow(edges, cmap=plt.cm.gray)

    row1, col1 = edges.shape
    for _, angle, dist in zip(*transform.hough_line_peaks(h, theta, d, min_distance=110, threshold=30, num_peaks=4)):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - col1 * np.cos(angle)) / np.sin(angle)
        ax1.plot((0, col1), (y0, y1), '-r')

    ax1.axis((0, col1, row1, 0))
    ax1.set_title('Detected lines')
    ax1.set_axis_off()
    plt.show()
