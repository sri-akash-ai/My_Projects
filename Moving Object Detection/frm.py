import cv2

vs = cv2.VideoCapture(0)  

while True:
    ret, img = vs.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("VideoStream", img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()

