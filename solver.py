import cv2
import numpy as np
import math
from PIL import Image

answers={
    0: {12:'male', 13:'female'},
    1: {xrange(4,6): 'fall', xrange(7,9): 'spring',xrange(10,12):'summer',},
    2: {1:'zzz', 2:'zzz',},
    3: {11:'strongly agree',12:'agree',13:'neutral',14:'disagree',15:'strongly disagree'}
}
questions={
    2:1,
    3:2,
    4:3,
    9:4,
    xrange(100, 102):5,
    xrange(104, 106):6,
    xrange(108, 110):7,
    xrange(112, 124):8,
    xrange(124, 126):9,
    xrange(128, 130):10,
    xrange(132, 134):11,
    xrange(136, 138):12,
    xrange(140, 142):13,
    xrange(144, 146):14,
    xrange(156, 158):15,
    xrange(160, 162):16,
    xrange(176, 178):17,
    xrange(176, 178):18,
    xrange(180, 182):19,
    xrange(189, 190):20,
    xrange(200, 202):21,
    xrange(204, 206):22,

}

def get_answers(centroids,length):
    answers=np.zeros((1,22),np.str)
    for i in range(0,21):
        if i==0:
            if centroids[0][1]<300:
                answers[0] = 'male' if (centroids[0][0]/100)%10== 2 else 'female'
            else:
                answers[0]='no answer'



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

def fit_image(img):
    cv2.imwrite('rotated.jpg', img)

    img_rotated = Image.open('rotated.jpg')

    # 1654 2338
    h = img.shape[0]
    w = img.shape[1]
    h1 = 2338
    w1 = 1654
    cropped_img = img_rotated.crop(((w - w1) // 2, (h - h1) // 2, (w + w1) // 2, (h + h1) // 2))

    cropped_img.save('rotated_new.jpg')

    return cv2.imread('rotated_new.jpg',0)


img=cv2.imread('tests/test_sample2.jpg',0)

#handle roation
angle=get_rotAngle(img)

if abs(angle)>0:
    img = rotate_image(img, angle)
    img=fit_image(img)

#removing all but circles
dummy, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
img=cv2.bitwise_not(img)

specific_area(img,465,100)



#1 1257 1390
# cv2.imshow('g',resize(img,35))
# cv2.waitKey(0)

#cv2.imwrite('rotated.jpg', rotated)


# specific_area(thresh)
# thresh=resize(thresh,35)
