import numpy as np
import cv2 as cv

# Apply CLAHE to colored image by converting to LAB colorspace
def applyCLAHE(img, clipLimit, tileGridSize):
    img_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    lab_channels = cv.split(img_lab)
    lab_channels[0] = clahe.apply(lab_channels[0])
    img_lab = cv.merge(lab_channels)
    img_clahe = cv.cvtColor(img_lab, cv.COLOR_LAB2BGR)

    return img_clahe


# Remove (as much as possible) non-grays from the image
def keepGrays(img):
    img_copy = img.copy()
    img_hsv = cv.cvtColor(img_copy, cv.COLOR_BGR2HSV)

    lower = np.array([0, 200, 40])
    upper = np.array([180, 255, 255])
    mask = cv.inRange(img_hsv, lower, upper)
    img_copy[mask > 0] = [255, 255, 255]

    return img_copy


for i in range(0, 26):
    img = cv.imread('images/{}.jpg'.format(i))
    img_clahe = applyCLAHE(img, 4.0, (8,8))
    img_masked = keepGrays(img_clahe)
    img_gray = cv.cvtColor(img_clahe, cv.COLOR_BGR2GRAY)
    img_thresh1 = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, 30)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))
    img_thresh1 = cv.morphologyEx(img_thresh1, cv.MORPH_OPEN, kernel)

    cnts, _ = cv.findContours(img_thresh1, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(cnts, key=cv.contourArea, reverse=True)
    mask = np.zeros(img.shape[:2], dtype="uint8")
    thresh1_copy = cv.cvtColor(img_thresh1, cv.COLOR_GRAY2BGR)
    for contour in contours:
        x,y,w,h = cv.boundingRect(contour)
        if w*h < 30:
            break
        elif h < 30 or (h*1.0/w) < 1.5:
            continue
        else:
            cv.rectangle(thresh1_copy, (x,y), (x+w,y+h), (0,0,255), 2)
            cv.drawContours(mask, [contour], -1, 1, -1)
        
    img_filtered_contours = cv.bitwise_and(img_thresh1, img_thresh1, mask=mask)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (31, 1))
    img_closed = cv.morphologyEx(img_filtered_contours, cv.MORPH_CLOSE, kernel)

    cnts, _ = cv.findContours(img_closed.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(cnts, key=cv.contourArea, reverse=True)

    for contour in contours:
        x,y,w,h = cv.boundingRect(contour)
        rbox = cv.minAreaRect(contour)
        pts = cv.boxPoints(rbox).astype(np.int32)
        cv.drawContours(img, [pts], -1, (0, 0, 255), 1, cv.LINE_AA)

    cv.imshow('Thresh 1', img_thresh1)
    cv.imshow('Thresh 1 Boxes', thresh1_copy)
    # cv.imshow('Filtered Contours', img_filtered_contours)
    # cv.imshow('Masked', img_masked)
    cv.imshow('Closed', img_closed)
    cv.imshow('Image', img)
    cv.imshow('Masked', img_masked)
    cv.waitKey(0)