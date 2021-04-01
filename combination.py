import cv2
from social_distancing import *
import numpy as np

birds_eye = cv2.imread('test_street_bird.jpg')
boxes_image = cv2.imread('test_street_boxes.jpg')
points = [[191, 487], [254, 388], [55, 387], [330, 370], [450, 330], [377, 274]]
birds, matrix = full_social_distancing(boxes_image, points, 80)

inv = np.linalg.inv(matrix)

inverted_image = cv2.warpPerspective(birds, inv, (960, 640))
added = cv2.add(boxes_image, inverted_image)
cv2.imshow('inverted', added)
cv2.imwrite('combined_street.jpg', added)

cv2.waitKey(0)