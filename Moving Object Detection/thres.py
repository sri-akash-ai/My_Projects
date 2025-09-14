import cv2

# Load image using full path
img = cv2.imread(r"d:\My_Projects\Moving Object Detection\novitech.png")

if img is None:
    print("Image not found!")
    exit()

grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresholdImg = cv2.threshold(grayImg, 170, 255, cv2.THRESH_BINARY)[1]

cv2.imshow("Original", img)
cv2.imshow("Threshold", thresholdImg)

cv2.waitKey(0)
cv2.destroyAllWindows()
