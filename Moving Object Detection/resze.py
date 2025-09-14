import cv2
import imutils

img = cv2.imread("d:/My_Projects/Moving Object Detection/novitech.png")


if img is None:
    print("Image not found!")
    exit()

resizedImg = imutils.resize(img, width=50)
cv2.imwrite('resizedImage2.jpg', resizedImg)
print("Image resized and saved as resizedImage2.jpg")
