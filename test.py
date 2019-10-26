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

def removeBadContours(img, minHeight = 25, maxHeight = 350, minRatio = 1.5):
    img = img.copy()
    img_colored = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    mask = np.zeros(img.shape[:2], dtype="uint8")

    cnts, _ = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(cnts, key=cv.contourArea, reverse=True)
    for contour in contours:
        x,y,w,h = cv.boundingRect(contour)

        if w*h < minHeight:     # exit loop if area is less than minHeight
            break
        elif h < minHeight or h > maxHeight or (h*1.0/w) < minRatio:    # if contour is bad, skip
            continue
        else:
            cv.drawContours(mask, [contour], -1, 1, -1)
            cv.rectangle(img_colored, (x,y), (x+w,y+h), (0,0,255), 2)

        img_filtered_contours = cv.bitwise_and(img, img, mask=mask)

    return img_filtered_contours, img_colored, mask

def removeBadTargets(img, minArea = 2000, maxRatioHorizontal = 3, maxRatioVertical = 2.5):
    img = img.copy()
    img_colored = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    mask = np.zeros(img.shape[:2], dtype="uint8")

    cnts, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(cnts, key=cv.contourArea, reverse=True)

    for contour in contours:
        rbox = cv.minAreaRect(contour)
        (x, y), (width, height), angle = rbox
        x,y,width,height = int(x), int(y), int(width), int(height)
        pts = cv.boxPoints(rbox).astype(np.int32)

        if -90.0 <= angle <= -45.0:     # Switch height and width if angle between -90 and -45 to keep logic below
            width,height = height,width

        if width*height < minArea:
            cv.drawContours(mask, [pts], -1, 0, -1)
            cv.drawContours(img_colored, [pts], -1, (0, 255, 0), 1, cv.LINE_AA)
        elif width*1.0/height > maxRatioHorizontal or height*1.0/width > maxRatioVertical:
            cv.drawContours(mask, [pts], -1, 0, -1)
            cv.drawContours(img_colored, [pts], -1, (255, 0, 0), 1, cv.LINE_AA)
        else:
            cv.drawContours(mask, [pts], -1, 1, -1)
            cv.drawContours(img_colored, [pts], -1, (0, 0, 255), 1, cv.LINE_AA)

    img_filtered_targets = cv.bitwise_and(img, img, mask=mask)

    return img_filtered_targets, img_colored, mask


for i in range(0, 26):
    img = cv.imread('images/{}.jpg'.format(i))
    img_clahe = applyCLAHE(img, 4.0, (8,8))
    img_gray = cv.cvtColor(img_clahe, cv.COLOR_BGR2GRAY)

    # Inverse image thresholding so our objects of interest (black bars) become white
    img_thresh = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, 30)

    # Noise removal
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))
    img_thresh = cv.morphologyEx(img_thresh, cv.MORPH_OPEN, kernel)

    img_filtered_contours, _, _ = removeBadContours(img_thresh)

    # Close black bars
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (31, 1))
    img_closed = cv.morphologyEx(img_filtered_contours, cv.MORPH_CLOSE, kernel)

    img_filtered_targets, img_colored, _ = removeBadTargets(img_closed)

    cv.imshow('CLAHE', img_clahe)
    cv.imshow('Gray', img_gray)
    cv.imshow('Thresh', img_thresh)
    cv.imshow('Filtered Contours', img_filtered_contours)
    cv.imshow('Closed', img_closed)
    cv.imshow('Filtered targets', img_filtered_targets)
    cv.imshow('Colored', img_colored)

    img_filtered_contours_copy = img_filtered_contours.copy()
    cnts, _ = cv.findContours(img_filtered_targets, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(cnts, key=cv.contourArea, reverse=True)
    for contour in contours:
        mask = np.zeros(img_filtered_contours_copy.shape[:2], dtype="uint8")
        rbox = cv.minAreaRect(contour)
        pts = cv.boxPoints(rbox).astype(np.int32)
        cv.drawContours(mask, [pts], -1, 1, -1)
        target = cv.bitwise_and(img_filtered_contours_copy, img_filtered_contours_copy, mask=mask)

        cv.imshow('Target', target)
        cv.waitKey(0)