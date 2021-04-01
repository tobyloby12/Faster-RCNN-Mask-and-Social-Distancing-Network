import cv2
from scipy import ndimage
import numpy as np

def warp_image(image, width, height):
    #defining target rectangle
    # tl = [1600-360, 1200-562]
    # tr = [1600-1262, 1200-563]
    # br = [1600-1190, 1200-993]
    # bl = [1600-446, 1200-977]
    # tl = [0, 0]
    # tr = [width, 0]
    # br = [width, height]
    # bl = [0, height]
    # tl = [75, 102]
    # tr = [388, 89]
    # br = [845, 292]
    # bl = [3, 435]
    # tl = [107, 275]
    # tr = [312, 249]
    # br = [461, 404]
    # bl = [110, 479]
    tl = [272,225]
    tr = [439,226]
    br = [773,419]
    bl = [6,401]
    target_rectangle = np.float32([tl,tr,br,bl])

    # defining image parameters

    imgtl = [0,0]
    imgtr = [0,width]
    imgbr = [height, width]
    imgbl = [height,0]
    img_params = np.float32([imgtl,imgtr,imgbr,imgbl])

    # creating perspective transform and applying
    matrix = cv2.getPerspectiveTransform(target_rectangle, img_params)
    img_transformed = cv2.warpPerspective(image, matrix, (width, height))
    return img_transformed, matrix

def point_warp(points, matrix):
    list_warped = list()
    list_points = np.float32(points).reshape(-1, 1, 2)
    warped_points = cv2.perspectiveTransform(list_points, matrix)
    for point in warped_points:
        list_warped.append(point)
    return list_warped

def euclidean_distance_calculator(point1, point2):
    x1 = point1[0][0]
    y1 = point1[0][1]
    x2 = point2[0][0]
    y2 = point2[0][1]
    distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    return distance

def social_distancing_determine(points, distance_per_pixel):
    MIN_DIST = 1.5
    min_dist = MIN_DIST*distance_per_pixel
    distancing = list()
    not_distancing = list()
    points_dict = {}
    for i in range(len(points)):
        points_dict[f'Point {i}'] = points[i]
    for i in points_dict:
        distancing.append(i)
        for j in points_dict:
            if i != j:
                distance = euclidean_distance_calculator(points_dict[i], points_dict[j])
                if distance < min_dist:
                    copyi = False
                    copyj = False
                    for k in not_distancing:
                        if k==i:
                            copyi = True
                        if k==j:
                            copyj = True
                    if copyi == False:
                        not_distancing.append(i)
                    if copyj == False:
                        not_distancing.append(j)
    for i in not_distancing:
        for j in distancing:
            if i==j:
                distancing.remove(i)

    return(distancing, not_distancing, points_dict)

def plot_birds_eye_view(warped_image, distancing, not_distancing, point_dict, width, height, ppm):
    # plotting social distancing
    warped_image = cv2.rectangle(warped_image, (0, 0), (width, height), color=(0, 0, 0), thickness=-1)
    for i in distancing:
        warped_image = cv2.circle(warped_image, (point_dict[i][0][0], point_dict[i][0][1]), radius=0, color=(0, 255, 0), thickness=5)
        warped_image = cv2.circle(warped_image, (point_dict[i][0][0], point_dict[i][0][1]), radius=int(ppm*1.5), color=(0, 255, 0), thickness=2)
    for i in not_distancing:
        warped_image = cv2.circle(warped_image, (point_dict[i][0][0], point_dict[i][0][1]), radius=0, color=(0, 0, 255), thickness=5)
        warped_image = cv2.circle(warped_image, (point_dict[i][0][0], point_dict[i][0][1]), radius=int(ppm*1.5), color=(0, 0, 255), thickness=2)
    return warped_image

def combining(warped_image, matrix, image, height, width):
    inv = np.linalg.inv(matrix)
    inverted_image = cv2.warpPerspective(warped_image, inv, (width, height))
    added = cv2.add(image, inverted_image)
    return added


# # # defining image and rotating it
# Image = cv2.imread('perspective image.jpeg')
# rotated = ndimage.rotate(Image, 90)
# # applying perspective warp
# height,width,depth = rotated.shape
# warped_image, warp_matrix = warp_image(rotated, width, height)

# # # warping points
# points = [[1600-540, 1200-700],[1600-771, 1200-762], [1600-846, 1200-762], [1600-1061, 1200-887]]
# warped_points = point_warp(points, warp_matrix)

# # finding which points are distancing
# distancing, not_distancing, point_dict = social_distancing_determine(warped_points, 150)

# # plotting social distancing
# for i in distancing:
#     warped_image = cv2.circle(warped_image, (point_dict[i][0][0], point_dict[i][0][1]), radius=0, color=(0, 255, 0), thickness=5)
#     warped_image = cv2.circle(warped_image, (point_dict[i][0][0], point_dict[i][0][1]), radius=200, color=(0, 255, 0), thickness=2)
# for i in not_distancing:
#     warped_image = cv2.circle(warped_image, (point_dict[i][0][0], point_dict[i][0][1]), radius=0, color=(0, 0, 255), thickness=5)
#     warped_image = cv2.circle(warped_image, (point_dict[i][0][0], point_dict[i][0][1]), radius=200, color=(0, 0, 255), thickness=2)

# cv2.imshow('Image', warped_image)
# cv2.imwrite('perspective_test_image.jpg', warped_image)
# cv2.waitKey(0)

def full_social_distancing(image, points, distance_per_pixel):
    height, width, depth = image.shape
    warped_image, warp_matrix = warp_image(image, width, height)
    warped_points = point_warp(points, warp_matrix)
    distancing, not_distancing, point_dict = social_distancing_determine(warped_points, distance_per_pixel)
    plot_birds_eye_view(warped_image, distancing, not_distancing, point_dict, width, height, distance_per_pixel)
    added = combining(warped_image,warp_matrix,image,height,width)
    return warped_image, added

# Image = cv2.imread('videos\masks\image_115.jpg')
# points = [[374, 412], [170, 421]]
# warped_image = full_social_distancing(Image, points, 460)
# cv2.imshow('Image', warped_image)
# cv2.imwrite('test_image.jpg', warped_image)
# cv2.waitKey(0)