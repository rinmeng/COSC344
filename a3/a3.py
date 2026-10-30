import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

bab = os.path.join(script_dir, "Baboon.png")
bab_gray = os.path.join(script_dir, "BaboonGray.png")


f = cv2.imread(bab_gray, cv2.IMREAD_GRAYSCALE)


# 1.a)
def imageScaling4e(f, cx, cy):
    M, N = f.shape

    M_out = round(cy * M)
    N_out = round(cx * N)

    g = np.zeros((M_out, N_out), dtype=f.dtype)

    y_out, x_out = np.indices((M_out, N_out))
    x_src = x_out / cx
    y_src = y_out / cy
    x_nn = np.floor(x_src + 0.5).astype(int)
    y_nn = np.floor(y_src + 0.5).astype(int)
    valid = (x_nn >= 0) & (x_nn < N) & (y_nn >= 0) & (y_nn < M)
    g[y_out[valid], x_out[valid]] = f[y_nn[valid], x_nn[valid]]

    return g


M, N = f.shape
cx = 2
cy = 1 / 2

f_translated = imageScaling4e(f, cx, cy)

plt.figure(figsize=(10, 5))
plt.title("Scaled image")
plt.imshow(f_translated, cmap="gray", vmin=0, vmax=255)
plt.axis("off")
plt.show()


# 1.b)
def imageRotate4e(f, theta, mode="crop"):
    if mode not in ("full", "crop"):
        raise ValueError("mode must be either full or crop")
    M, N = f.shape

    theta_rad = np.deg2rad(theta)
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)

    if mode == "crop":
        M_out, N_out = M, N
    else:
        N_out = int(np.ceil(N * abs(c) + M * abs(s)))
        M_out = int(np.ceil(N * abs(s) + M * abs(c)))

    g = np.zeros((M_out, N_out), dtype=f.dtype)

    y_out, x_out = np.indices((M_out, N_out))

    cx_in = (N - 1) / 2
    cy_in = (M - 1) / 2
    cx_out = (N_out - 1) / 2
    cy_out = (M_out - 1) / 2

    xo = x_out - cx_out
    yo = y_out - cy_out

    x_src = c * xo - s * yo + cx_in
    y_src = s * xo + c * yo + cy_in

    x_nn = np.floor(x_src + 0.5).astype(int)
    y_nn = np.floor(y_src + 0.5).astype(int)

    valid = (x_nn >= 0) & (x_nn < N) & (y_nn >= 0) & (y_nn < M)
    g[y_out[valid], x_out[valid]] = f[y_nn[valid], x_nn[valid]]

    return g


theta = 45
f_rotated_full = imageRotate4e(f, theta, "full")
f_rotated_crop = imageRotate4e(f, theta)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Rotated image (crop)")
plt.imshow(f_rotated_crop, cmap="gray", vmin=0, vmax=255)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Rotate image (full)")
plt.imshow(f_rotated_full, cmap="gray", vmin=0, vmax=255)
plt.axis("off")

plt.show()
