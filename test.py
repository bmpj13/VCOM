import numpy as np
import cv2 as cv
from math import ceil

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """
    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def get_box_values(box):
    (x, y), (width, height), angle = box
    x, y, width, height = x, y, width, height

    if angle < -45.0:     # Switch height and width if angle between -90 and -45 ( width, height 90º = height, width 0º )
        width, height = height, width

    return x, y, width, height, angle

# Apply CLAHE to colored image by converting to LAB colorspace
def applyCLAHE(img, clipLimit, tileGridSize):
    img_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    lab_channels = cv.split(img_lab)
    lab_channels[0] = clahe.apply(lab_channels[0])
    img_lab = cv.merge(lab_channels)
    img_clahe = cv.cvtColor(img_lab, cv.COLOR_LAB2BGR)

    return img_clahe

def removeBadContours(img, minHeight = 20, maxHeight = 350, minRatio = 2):
    img = img.copy()
    img_colored = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    mask = np.zeros(img.shape[:2], dtype="uint8")
    
    cnts, _ = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(cnts, key=cv.contourArea, reverse=True)
    img_filtered_contours = np.zeros(img.shape[:2], dtype="uint8")
    for contour in contours:
        rbox = cv.minAreaRect(contour)
        pts = cv.boxPoints(rbox).astype(np.int32)
        x, y, width, height, _ = get_box_values(rbox)

        if width * height < minHeight:     # exit loop if area is less than minHeight
            break
        elif height < minHeight or height > maxHeight or (height * 1.0 / width) < minRatio:    # if contour is bad, skip
            continue
        else:
            cv.drawContours(mask, [contour], -1, 1, -1)
            cv.drawContours(img_colored, [pts], -1, (0, 255, 0), 1, cv.LINE_AA)

        img_filtered_contours = cv.bitwise_and(img, img, mask=mask)

    return img_filtered_contours, img_colored, mask

def removeBadTargets(img, minArea = 2000, maxRatioHorizontal = 3, maxRatioVertical = 2):
    img = img.copy()
    img_colored = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    mask = np.zeros(img.shape[:2], dtype="uint8")

    cnts, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(cnts, key=cv.contourArea, reverse=True)
    for contour in contours:
        rbox = cv.minAreaRect(contour)
        pts = cv.boxPoints(rbox).astype(np.int32)
        x, y, width, height, _ = get_box_values(rbox)

        if width * height < minArea:
            break
        elif width * 1.0 / height > maxRatioHorizontal or height * 1.0 / width > maxRatioVertical:
            continue
        else:
            cv.drawContours(mask, [pts], -1, 1, -1)
            cv.drawContours(img_colored, [pts], -1, (0, 255, 0), 1, cv.LINE_AA)

    img_filtered_targets = cv.bitwise_and(img, img, mask=mask)

    return img_filtered_targets, img_colored, mask

def pickROIs(img_src, img_dest):
    img_src = img_src.copy()
    img_dest = img_dest.copy()
    rois = []

    cnts, _ = cv.findContours(img_src, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(cnts, key=cv.contourArea, reverse=True)
    for contour in contours:
        mask = np.zeros(img_src.shape[:2], dtype="uint8")
        rbox = cv.minAreaRect(contour)
        x, y, width, height, angle = get_box_values(rbox)

        if angle < -45.0:
            angle += 90.0

        pts = cv.boxPoints(rbox).astype(np.int32)
        cv.drawContours(mask, [pts], -1, 1, -1)

        img_target = cv.bitwise_and(img_dest, img_dest, mask=mask)
        img_target = rotate_image(img_target, angle)
        img_thinned = cv.Canny(img_target, 0, 150, apertureSize=3, L2gradient=True)

        hough_lines = cv.HoughLines(img_thinned, 1, np.pi/6, 20, max_theta=np.pi/4)
        hough_lines = hough_lines[:, 0, :] if hough_lines is not None else []

        if len(hough_lines) >= 15:
            rois.append(mask)
    
    return rois

def getBarcodes(img, angle = 0):
    img_rotated = rotate_image(img, angle)
    img_clahe = applyCLAHE(img_rotated, 4.0, (8,8))
    img_gray = cv.cvtColor(img_clahe, cv.COLOR_BGR2GRAY)

    # Inverse image thresholding so our objects of interest (black bars) become white
    img_thresh = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, 30)

    # Noise removal
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))
    img_thresh = cv.morphologyEx(img_thresh, cv.MORPH_OPEN, kernel)

    img_filtered_contours, _, _ = removeBadContours(img_thresh)

    # Close black bars
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (31, 11))
    img_closed = cv.morphologyEx(img_filtered_contours, cv.MORPH_CLOSE, kernel)

    img_filtered_targets, _, _ = removeBadTargets(img_closed)

    rois = pickROIs(img_filtered_targets, img_filtered_contours)

    for i, roi in enumerate(rois):
        rois[i] = rotate_image(roi, -angle)

    cv.imshow('Original', img)
    cv.imshow('CLAHE', img_clahe)
    cv.imshow('Gray', img_gray)
    cv.imshow('Threshold', img_thresh)
    cv.imshow('Countours Filter', img_filtered_contours)
    cv.imshow('Closed', img_closed)
    cv.imshow('Targets Filter', img_filtered_targets)

    return rois

for i in range(0, 27):
    img = cv.imread('images/{}.jpg'.format(i))
    barcodes = getBarcodes(img) # + getBarcodes(img, 90)
    
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for roi in barcodes:
        mask = cv.bitwise_or(mask, roi)
    cv.imshow('Barcodes', cv.bitwise_and(img, img, mask=mask))
    cv.waitKey(0)
