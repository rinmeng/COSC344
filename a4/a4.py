# Rin Meng 51940633

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
bab = os.path.join(script_dir, "Baboon.png")
bab_gray = os.path.join(script_dir, "BaboonGray.png")


# 1. Image histogram
img = cv2.imread(bab_gray, cv2.IMREAD_GRAYSCALE)

plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.title("click top left corner then bottom right corner")
pts = plt.ginput(2, timeout=0)
plt.close()

(xa, ya), (xb, yb) = pts
x1, x2 = sorted([int(xa), int(xb)])
y1, y2 = sorted([int(ya), int(yb)])

result = img.copy()
result[y1:y2, x1:x2] = cv2.equalizeHist(img[y1:y2, x1:x2])

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(result, cmap="gray", vmin=0, vmax=255)
plt.title("ROI equalized")

plt.show()


A = np.array(
    [
        [5, 8, 3, 4, 6, 2, 3, 7],
        [3, 2, 1, 1, 9, 5, 1, 0],
        [0, 9, 5, 3, 0, 4, 8, 3],
        [4, 2, 7, 2, 1, 9, 0, 6],
        [9, 7, 9, 8, 0, 4, 2, 4],
        [5, 2, 1, 8, 4, 1, 0, 9],
        [1, 8, 5, 4, 9, 2, 3, 8],
        [3, 7, 1, 2, 3, 4, 4, 6],
    ]
)

B = np.array(
    [
        [2, 1, 0],
        [1, 1, -1],
        [0, -1, -2],
    ]
)


# 2. Convolution
def imageConv(f, h, mode="zero"):
    if mode not in ("zero", "replicate"):
        raise ValueError("mode must be 'zero' or 'replicate'")

    f = np.asarray(f, dtype=float)
    h = np.asarray(h, dtype=float)
    h_flipped = np.flip(h)

    kh, kw = h.shape
    ph, pw = kh // 2, kw // 2

    if mode == "zero":
        f_padded = np.pad(f, ((ph, ph), (pw, pw)), mode="constant")
    else:
        f_padded = np.pad(f, ((ph, ph), (pw, pw)), mode="edge")

    res = np.zeros(f.shape)
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            window = f_padded[i : i + kh, j : j + kw]
            res[i, j] = np.sum(window * h_flipped)

    return res


print("A * B (zero padding):")
print(imageConv(A, B).astype(int))
print("A * B (replicate padding):")
print(imageConv(A, B, mode="replicate").astype(int))
