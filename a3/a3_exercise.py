import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

bab = os.path.join(script_dir, "Baboon.png")
bab_gray = os.path.join(script_dir, "BaboonGray.png")


f = cv2.imread(bab_gray, cv2.IMREAD_GRAYSCALE)


def imageTranslate4e(f, tx, ty, mode="black"):
    if mode not in ("black", "white"):
        raise ValueError("mode must be 'black' or 'white'")
    M, N = f.shape  # rows (y), cols (x)

    bg = 0 if mode == "black" else 255

    g = np.full((M, N), bg, dtype=f.dtype)

    y_out, x_out = np.indices((M, N))

    x_src = x_out - tx
    y_src = y_out - ty

    x_nn = np.floor(x_src + 0.5).astype(int)
    y_nn = np.floor(y_src + 0.5).astype(int)

    valid = (x_nn >= 0) & (x_nn < N) & (y_nn >= 0) & (y_nn < M)

    g[y_out[valid], x_out[valid]] = f[y_nn[valid], x_nn[valid]]

    return g


M, N = f.shape

tx = N / 4
ty = M / 2


g_black = imageTranslate4e(f, tx, ty)
g_white = imageTranslate4e(f, tx, ty, "white")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(g_black, cmap="gray", vmin=0, vmax=255)
plt.title("first translated image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(g_white, cmap="gray", vmin=0, vmax=255)
plt.title("second translated image")
plt.axis("off")

plt.show()
