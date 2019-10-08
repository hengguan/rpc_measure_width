import glob
import cv2
import pickle
import numpy as np
from scipy.misc import imshow
from PIL import Image
from scipy.spatial import distance as dist


img_shape = None
pixelsPerMetric = None

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def is_vertical(line):
    x1, y1, x2, y2 = line
    if y2-y1 == 0:
        return False
    t = (x2-x1)/(y2-y1)
    if t<0.5 and t>-0.5:
        return True
    return False

def is_horizon(line):
    x1, y1, x2, y2 = line
    if x2-x1 == 0:
        return False
    t = (y2-y1)/(x2-x1)
    if t<0.5 and t>-0.5:
        return True
    return False
def display_result(img, line_left, line_right, mid1, mid2, pixel_dist):
    global pixelsPerMetric
    x1, y1, x2, y2 = line_left
    x3, y3, x4, y4 = line_right
    
    '''if pixelsPerMetric is None:
        pixelsPerMetric = float(pixel_dist) / 24.65'''
    res_dist = pixel_dist # float(pixel_dist)/pixelsPerMetric
    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
    cv2.line(img, (int(x3), int(y3)), (int(x4), int(y4)), (255, 0, 255), 2)
    cv2.line(img, (int(mid1[0]), int(mid1[1])), (int(mid2[0]), int(mid2[1])), (255, 0, 255), 2)
    cv2.circle(img, (int(mid1[0]), int(mid1[1])), 5, (255, 0, 0), -1)
    cv2.circle(img, (int(mid2[0]), int(mid2[1])), 5, (255, 0, 0), -1)
    cv2.putText(img, "{:.2f}mm".format(res_dist),
            (int(mid2[0] - 15), int(mid2[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2)
    # cv2.rectangle(5mg, (int(m1[0]), int(m1[1])), (int(m2[0]), int(m2[1])), (0, 0, 255), 2)
    # imshow(img)

    return res_dist

def pixel_to_camera_frame(pixel, intr_mat):
    line_coef = [3.71315216e-05, -6.83351876e-05,  1.09282172e-03]
    # line_coef = [i*0.985 for i in line_coef]
    length = np.sqrt(line_coef[0]**2+line_coef[1]**2+line_coef[2]**2)
    xc_no_zc = (pixel[0]-intr_mat[0][2])/intr_mat[0][0]
    yc_no_zc = (pixel[1]-intr_mat[1][2])/intr_mat[1][1]
    zc = (1-length*0.64)/(xc_no_zc*line_coef[0]+yc_no_zc*line_coef[1]+line_coef[2])
    xc = xc_no_zc*zc
    yc = yc_no_zc*zc
    return (xc, yc, zc)


def create_rect(img, direct, line1, line2, gt, intr_mat):
    global pixelsPerMetric
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    m1 = midpoint(line1[:2], line1[2:])
    m2 = midpoint(line2[:2], line2[2:])
    if ((x2==x1) and (x4==x3)) or direct==0:
        p1_3d = pixel_to_camera_frame(line1[2:], intr_mat)
        p2_3d = pixel_to_camera_frame(line2[2:], intr_mat)
        mid_y = m1[1]+(m2[1]-m1[1])/2.0
        pixel_dist = dist.euclidean(p1_3d, p2_3d) #x4 - x2  

        return display_result(
            img, 
            (m1[0], m1[1], m1[0], m2[1]),
            (m2[0], m2[1], m2[0], m1[1]),
            (m1[0], mid_y),
            (m2[0], mid_y),pixel_dist)
    k = -1 / direct # (k5+k2)*0.5
    # m1 l1
    l1_x = (m2[1]-m1[1]+k*m1[0]+m2[0]/k)/(k+1/k)
    l1_y = (k*m2[1]+m1[1]/k+m2[0]-m1[0])/(k+1/k)
    # m2 l2
    l2_x = (m1[1]-m2[1]+k*m2[0]+m1[0]/k)/(k+1/k)
    l2_y = (k*m1[1]+m2[1]/k+m1[0]-m2[0])/(k+1/k)

    mid1 = midpoint((m1[0], m1[1]),(l1_x, l1_y))
    mid2 = midpoint((m2[0], m2[1]), (l2_x, l2_y))
    p1_3d = pixel_to_camera_frame(mid1, intr_mat)
    p2_3d = pixel_to_camera_frame(mid2, intr_mat)
    pixel_dist = dist.euclidean(p1_3d, p2_3d)

    return display_result(
        img, (m1[0], m1[1], l1_x, l1_y),
        (m2[0], m2[1], l2_x, l2_y),
        (mid1[0], mid1[1]),(mid2[0], mid2[1]), pixel_dist)


def line_detection_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)  
    print(lines)
    for line in lines:
        rho, theta = line[0]  
        a = np.cos(theta)   
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))  
        y1 = int(y0 + 1000 * (a))  
        x2 = int(x0 - 1000 * (-b))  
        y2 = int(y0 - 1000 * (a))   
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # cv2.imshow("image_lines", image)
    imshow(image)


def line_detect_possible_demo(image, gt, intr_mat):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    gray = image[:, :, 0] # cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    ret, binary = cv2.threshold(gray,80,255,cv2.THRESH_OTSU)
    binary[:, :20] = 255
    binary[:20, :] = 255
    binary[-20:, :] = 255
    binary[:, -20:] = 255
    # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 5)
    edges = cv2.Canny(binary, 50, 150)
    # imshow(binary)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 70, minLineLength=50, maxLineGap=20)
    # print(lines)
    left = (image.shape[0], image.shape[1])
    right = (0, image.shape[1])
    k = None
    lines = [l[0] for l in lines]
    def _bottom_up(elem):
        m = midpoint(elem[:2], elem[2:])
        return m[1]
    lines.sort(key=_bottom_up, reverse=True)
    for line in lines:
        x1, y1, x2, y2 = line
        midP = midpoint((x1, y1), (x2, y2))
        line_length = dist.euclidean((x1, y1), (x2, y2))
        if is_horizon(line) and line_length > 150 and midP[1]<(image.shape[0]-21):
            # cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
            k = (k+(y2-y1)/(x2-x1))*0.5 if k is not None else (y2-y1)/(x2-x1)

        if is_vertical(line):
            # cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
            
            if midP[0]< (left[0]+50) and midP[1]<left[1]:
                left = midP
                left_line = line
            if midP[0]>(right[0]-50) and midP[1]<right[1]:
                right = midP
                right_line = line

    '''x1, y1, x2, y2 = left_line
    cv2.line(image, (x1, y1), (x2, y2), (128, 0, 255), 2)
    x1, y1, x2, y2 = right_line
    cv2.line(image, (x1, y1), (x2, y2), (255, 128, 0), 2)
    imshow(image)'''
    return create_rect(image, k, left_line, right_line, gt, intr_mat)  


if __name__ == "__main__":
    imgs = []
    # for i in range(1, 5):
    # imgs += glob.glob('test2/*.bmp')
    with open('test2.txt', 'r') as f:
        imgs = [line.split(' ') for line in f.readlines()]
    print(len(imgs))
    with open('wide_dist_pickle.pk', 'rb') as f:
        mat = pickle.load(f)
    mtx = mat["mtx"]
    dst = mat["dist"]
    err = 0
    for im in imgs:
        im_name = im[0]
        gt = float(im[1])
        img = Image.open('test2/'+im_name).convert('RGB')
        # img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img = np.asarray(img, dtype=np.uint8)
        img = cv2.undistort(img, mtx, dst, None, mtx)
        # img = cv2.imread(im)
        pred = line_detect_possible_demo(img, gt, mtx)
        print(pred, gt)
        err += np.abs(pred-gt)
        
        # line_detection_demo(img)
        cv2.imshow("input image", img)
        cv2.waitKey(1000)
    print('mean error: {}'.format(err/11.0))
    # img = cv2.imread("gh/6/Image__2019-09-16__09-52-53.bmp")
    # cv2.namedWindow("input image", cv2.WINDOW_AUTOSIZE)
    
    # 
    # line_detect_possible_demo(img)
    # line_detection_demo(img)
    cv2.destroyAllWindows()