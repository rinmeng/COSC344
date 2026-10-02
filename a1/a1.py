import os

import cv2
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

bab = os.path.join(script_dir, "Baboon.png")
bab_gray = os.path.join(script_dir, "BaboonGray.png")


# question a
def pixVal4e(f, r, c):
    assert f is not None
    v = f[r, c]
    return v


# question b
f_read = cv2.imread(bab_gray, cv2.IMREAD_GRAYSCALE)
assert f_read is not None
print("Origin:", pixVal4e(f_read, 0, 0))
print("Image Center:", pixVal4e(f_read, f_read.shape[0] // 2, f_read.shape[1] // 2))


# question c
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


# question d
result = cursorValues4e(f_read)
if result is not None:
    r, c, v = result
    print(f"Center of right pupil: (r={r}, c={c}), value={v}")
