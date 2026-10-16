import os

import cv2
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

cameraman = os.path.join(script_dir, "cameraman.png")

A = cv2.imread(cameraman, cv2.IMREAD_GRAYSCALE)
B = cv2.imread(cameraman, cv2.IMREAD_GRAYSCALE)


sum_AB = cv2.add(A, B)
avg1 = cv2.divide(sum_AB, 2)
avg2 = cv2.add(cv2.divide(A, 2), cv2.divide(B, 2))
diff = cv2.absdiff(avg1, avg2)


titles = ["Original A", "(A + B) / 2", "A/2 + B/2", "|Difference|"]
images = [A, avg1, avg2, diff]

plt.figure(figsize=(14, 4))
for i, (img, t) in enumerate(zip(images, titles)):
    plt.subplot(1, 4, i + 1)
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.title(t)
    plt.axis("off")
plt.tight_layout()
plt.show()


"""
The two formulas are mathematically equal, but images are stored as uint8 (0 to 255).
In (A + B)/2, the sum exceeds 255 for any pixel ≥ 128 and is saturated to 255 before dividing,
so all bright pixels become about 128 and detail is lost. In A/2 + B/2,
each term is halved first so the sum stays within range,
no overflow occurs, and the result matches the original (within ±1 rounding error).
"""
