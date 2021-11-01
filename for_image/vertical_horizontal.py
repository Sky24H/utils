import cv2


def find_vertical(vertical):
    verticalsize = vertical.shape[0]//30
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, np.ones((3,3), np.uint8), iterations=1)
    return vertical

def find_horizontal(horizontal):
    horizontalsize = horizontal.shape[0]//30
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, np.ones((3,3), np.uint8), iterations=1)
    return horizontal
