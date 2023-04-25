import cv2
import numpy as np

img = cv2.imread("input.png")

b, g, r = cv2.split(img)

cv2.imwrite("temp/input_b.png", b)
cv2.imwrite("temp/input_g.png", g)
cv2.imwrite("temp/input_r.png", r)


def blend_scale_by2(img):
    img = img.astype(int)
    for _ in range(2):
        e = img[0::2]
        o = img[1::2]
        img = (e + o) // 2
        img = np.transpose(img)
    return img

b_scaled = b[::]
for _ in range(3):
    b_scaled = blend_scale_by2(b_scaled)
    
cv2.imwrite("temp/b_scaled.png", b_scaled.astype(np.uint8))