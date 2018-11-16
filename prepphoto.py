import numpy as np
import cv2

img_temp = cv2.imread('Graphics/pic.png')
_, img = cv2.threshold(img_temp,128, 255,cv2.THRESH_BINARY)
cv2.imwrite('Graphics/stan_lee.png', img)


