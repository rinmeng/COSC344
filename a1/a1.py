import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
bab = os.path.join(script_dir, "Baboon.png")
bab_gray = os.path.join(script_dir, "BaboonGray.png")

# 1.a
img = cv2.imread(bab)
assert img is not None


def scanLine4e(f, l, loc):
    if loc != "row" and loc != "col":
        return None

    return f[l, :].copy() if loc == "row" else f[:, l].copy()


# 1.b
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mid = gray.shape[0] // 2
s = scanLine4e(gray, mid, "row")
assert s is not None

plt.figure()
plt.plot(s)
plt.title("Scan Line Plot")
plt.xlabel("Pixel Index")
plt.ylabel("Intensity")
plt.show()


# 2.a
def mask4e(M, N, rUL, cUL, rLR, cLR):
    if rUL > rLR or cUL > cLR:
        raise ValueError("Upper-left must be above and left of lower-right")
    if rUL < 0 or cUL < 0 or rLR >= M or cLR >= N:
        raise ValueError(f"Region exceeds {M} x {N} mask dimensions")
    mask = np.zeros((M, N), dtype=np.uint8)
    mask[rUL : rLR + 1, cUL : cLR + 1] = 1
    return mask


# 2.b
img_gray = cv2.imread(bab_gray, cv2.IMREAD_GRAYSCALE)
assert img_gray is not None
M, N = img_gray.shape
h, w = M // 2, N // 2
rUL = (M - h) // 2
cUL = (N - w) // 2
rLR = rUL + h - 1
cLR = cUL + w - 1
mask = mask4e(M, N, rUL, cUL, rLR, cLR)

# 2.c
count = int(np.sum(mask))
expected = h * w
print(f"Image size: {M} x {N}")
print(
    f"Mask 1-valued elements: {count}, expected: {expected} which is",
    "correct" if count == expected else "incorrect",
)
