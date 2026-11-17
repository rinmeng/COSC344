import numpy as np

A = np.array([0, 1, 2, 3, 2, 1, 0], dtype=float)

B = np.array([1, 3, -1], dtype=float)

B_flipped = np.flip(B)

pad = len(B) // 2

A_padded = np.pad(A, pad)


res = np.array([])

for i in range(len(A)):
    res = np.append(
        res,
        (A_padded[i] * B_flipped[0])
        + (A_padded[i + 1] * B_flipped[1])
        + (A_padded[i + 2] * B_flipped[2]),
    )

print(res)

print(np.convolve(A, B, mode="same"))
