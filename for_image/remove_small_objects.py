import cv2


def remove_small_objects(masks, threshold):
    cnts = cv2.findContours(masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < threshold:
            cv2.drawContours(masks, [c], -1, (0, 0, 0), -1)
    return masks
