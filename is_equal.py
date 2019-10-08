import cv2
import glob
import matplotlib.pyplot as plt
from scipy.misc import imshow


def is_equal(img):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    gray = img[:, :, 0] # cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    ret, binary = cv2.threshold(gray,80,255,cv2.THRESH_OTSU)
    # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 5)
    edges = cv2.Canny(binary, 50, 150)
    imshow(edges)
    '''plt.imshow(binary)
    plt.show()
    cv2.imshow('is equal', edges)
    cv2.waitKey(500)'''


if __name__ == '__main__':
    imgs = glob.glob('data/*.bmp')
    print(imgs)
    for im in imgs:
        img = cv2.imread(im)
        is_equal(img)
