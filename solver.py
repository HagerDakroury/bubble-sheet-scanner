import cv2
import numpy as np
import math

def get_answers(centroids):


def resize (img,percentage):
    scale_percent = percentage  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

def specific_area (image,area,error):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype('uint8')
    components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]
    components = components

    size = area
    img2 = np.zeros(image.shape)


    for i in range(1, components):
        if sizes[i] >size-error and sizes[i] <size+error :
            print(centroids[i])

def rotate_image(mat, angle):
  # angle in degrees

    height, width = mat.shape[:2]
    image_center = (width/2, height/2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def get_rotAngle(img):
    img_edges = cv2.Canny(img, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

    angles = []

    for x1, y1, x2, y2 in lines[0]:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    angle = np.median(angles)

    return angle


img=cv2.imread('tests/test_sample8.jpg',0)

#handle roation
angle=get_rotAngle(img)

if angle>0:
    img = rotate_image(img, angle)

#removing all but circles
#dummy, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
img=cv2.bitwise_not(img)

#specific_area(img,465,10)

cv2.imshow('g',resize(img,35))
cv2.waitKey(0)

#cv2.imwrite('rotated.jpg', rotated)


# specific_area(thresh)
# thresh=resize(thresh,35)
