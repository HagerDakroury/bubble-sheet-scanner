import cv2
import numpy as np

def resize (img,percentage):
    scale_percent = percentage  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def specific_area (image):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype('uint8')
    components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]
    components = components

    size = 465
    img2 = np.zeros(image.shape)

    for i in range(1, components):
        if sizes[i] >10 and sizes[i] <size+1 :
            print(centroids[i])



img=cv2.imread('tests/test_sample2.jpg',0)
dummy, thresh = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)

thresh=cv2.bitwise_not(thresh)

# specific_area(thresh)
thresh=resize(thresh,35)


cv2.imshow('test',thresh)
cv2.waitKey()
