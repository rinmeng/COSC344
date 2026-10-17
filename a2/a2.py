import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

bab = os.path.join(script_dir, "BaboonGray.png")

## A1 helpers


def mask4e(M, N, rUL, cUL, rLR, cLR):
    if rUL > rLR or cUL > cLR:
        raise ValueError("Upper-left must be above and left of lower-right")
    if rUL < 0 or cUL < 0 or rLR >= M or cLR >= N:
        raise ValueError(f"Region exceeds {M} x {N} mask dimensions")
    mask = np.zeros((M, N), dtype=np.uint8)
    mask[rUL : rLR + 1, cUL : cLR + 1] = 1
    return mask


def pixVal4e(f, r, c):
    assert f is not None
    v = f[r, c]
    return v


def cursorValues4e(f):
    plt.imshow(f, cmap="gray")
    pts = plt.ginput(n=1)
    if not pts:
        return None

    x, y = pts[0]
    r = round(y)
    c = round(x)
    v = pixVal4e(f, r, c)

    print(f"Row: {r}, Col: {c}, Value: {v}")
    return r, c, v


# 1a.
def imArithmetic4e(f1, f2, op):
    f1 = np.asarray(f1, dtype=np.float64)
    f2 = np.asarray(f2, dtype=np.float64)

    match op:
        case "add":
            g = cv2.add(f1, f2)
        case "subtract":
            g = cv2.subtract(f1, f2)
        case "multiply":
            g = cv2.multiply(f1, f2)
        case "divide":
            g = cv2.divide(f1, f2)
        case _:
            raise ValueError("op must be 'add', 'subtract', 'multiply', or 'divide'")
    return g


# 1b.
def q1b(use_cursor=False):
    f = cv2.imread(bab, cv2.IMREAD_GRAYSCALE)
    assert f is not None, f"Could not read {bab}"
    M, N = f.shape

    if use_cursor:
        print("Click the UPPER-LEFT corner of the ROI")
        rUL, cUL, _ = cursorValues4e(f)
        print("Click the LOWER-RIGHT corner of the ROI")
        rLR, cLR, _ = cursorValues4e(f)
    else:
        rUL, cUL, rLR, cLR = 40, 120, 250, 390

    m = mask4e(M, N, rUL, cUL, rLR, cLR)
    g = imArithmetic4e(f, m, "multiply")

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(f, cmap="gray", vmin=0, vmax=255)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(g, cmap="gray", vmin=0, vmax=255)
    plt.title("User-defined ROI")
    plt.axis("off")

    plt.show()


q1b(True)


# 2a.


cameraman = os.path.join(script_dir, "cameraman.png")

f = cv2.imread(cameraman, cv2.IMREAD_GRAYSCALE)


def brightnessCorr(f, percent, op):
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")

    f = np.asarray(f, dtype=np.uint8)

    match op:
        case "brighten":
            factor = 1 + percent / 100
        case "darken":
            factor = 1 - percent / 100
        case _:
            raise ValueError("op must be 'brighten' or 'darken'")

    # uint8 multiply saturates, so results are clipped to [0, 255]
    g = cv2.multiply(f, np.ones_like(f), scale=factor)
    return g


# 2b

g = brightnessCorr(f, 30, "brighten")


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(f, cmap="gray", vmin=0, vmax=255)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(g, cmap="gray", vmin=0, vmax=255)
plt.title("Brightened 30%")
plt.axis("off")

plt.show()


# 3a

logo = os.path.join(script_dir, "small_ubc_logo.jpg")


def insert(f1, f2, x, y):
    if f1 is None or f2 is None:
        raise ValueError("must have f1 or f2")

    if f1.ndim == 3:
        f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    if f2.ndim == 3:
        f2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

    h1, w1 = f1.shape[:2]
    h2, w2 = f2.shape[:2]
    if x < 0 or y < 0 or y + h1 > h2 or x + w1 > w2:
        raise ValueError(
            f"f1 ({h1}x{w1}) at (x={x}, y={y}) does not fit inside f2 ({h2}x{w2})"
        )

    g = f2.copy()
    roi = g[y : y + h1, x : x + w1]
    g[y : y + h1, x : x + w1] = cv2.add(roi, f1)
    return g


# 3b

f1 = cv2.imread(logo, cv2.IMREAD_GRAYSCALE)
f2 = cv2.imread(cameraman, cv2.IMREAD_GRAYSCALE)


g = insert(f1, f2, 10, 10)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(f2, cmap="gray", vmin=0, vmax=255)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(g, cmap="gray", vmin=0, vmax=255)
plt.title("Modified Image")
plt.axis("off")

plt.show()
