import cv2


def visualize(image):
    if image.dtype != np.uint8:
        image *= 255
        image[image < 0] = 0
        image = image.astype(np.uint8)
    image = cv2.resize(image, (500, 500))
    cv2.imshow("Pressure", image)
    if cv2.waitKey(1) & 0xff == 27:
        return False
    else:
        return True
