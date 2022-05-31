import glob

import cv2

imgs = glob.glob("./trainset/*.jpg")
for i in imgs:
    mat = cv2.imread(i, cv2.IMREAD_UNCHANGED)
    b, g, r = cv2.split(mat)
    merged = cv2.merge([b,b,g,g,r,b,b,r])
    cv2.imwrite("1.bmp", merged)
    break