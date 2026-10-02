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
