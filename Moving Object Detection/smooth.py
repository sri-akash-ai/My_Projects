import cv2

# Load image using full path
img = cv2.imread(r"d:\My_Projects\Moving Object Detection\novitech.png")

if img is None:
    print("Image not found!")
    exit()

gaussianImg = cv2.GaussianBlur(img, (41, 41), 0)
gaussianImg1 = cv2.GaussianBlur(img, (21, 21), 10)

cv2.imshow("GaussianBlur", gaussianImg)
cv2.imshow("GaussianBlur1", gaussianImg1)

cv2.waitKey(0)
cv2.destroyAllWindows()

