import numpy as np
import cv2 as cv

for i in range(0, 26):
    img = cv.imread('images/{}.jpg'.format(i))
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_gray)

    img_thresh1 = cv.adaptiveThreshold(img_clahe, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 13, 35)
    img_blurred = cv.blur(img_thresh1, (25,25))
    _, img_thresh2 = cv.threshold(img_blurred, 55, 255, cv.THRESH_BINARY)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (30, 1))
    closed = cv.morphologyEx(img_thresh2, cv.MORPH_CLOSE, kernel)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    morphed = cv.erode(closed, kernel, iterations=4)
    morphed = cv.dilate(morphed, kernel, iterations=4)

    cnts, _ = cv.findContours(morphed.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(cnts, key=cv.contourArea, reverse=True)

    for contour in contours:
        rbox = cv.minAreaRect(contour)
        pts = cv.boxPoints(rbox).astype(np.int32)
        cv.drawContours(img, [pts], -1, (0, 255, 0), 1, cv.LINE_AA)

    # cv.imshow('Gray', img_gray)
    # cv.imshow('Threshold 1', img_thresh1)
    # cv.imshow('Blurred', img_blurred)
    # cv.imshow('Threshold 2', img_thresh2)
    # cv.imshow('Closed', closed)
    cv.imshow('Morphed', morphed)
    cv.imshow('Image', img)

    cv.waitKey(0)

cv.destroyAllWindows()



# def plotHoughLines(img, lines, color=(0, 0, 255)):
#     for rho, theta in lines:
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a*rho
#         y0 = b*rho
#         x1 = int(x0 + 1000*(-b))
#         y1 = int(y0 + 1000*(a))
#         x2 = int(x0 - 1000*(-b))
#         y2 = int(y0 - 1000*(a))
#         cv.line(img, (x1, y1), (x2, y2), color, 1)


# # Apply CLAHE to colored image by converting to LAB colorspace
# def applyCLAHE(img, clipLimit, tileGridSize):
#     img_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
#     clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

#     lab_channels = cv.split(img_lab)
#     lab_channels[0] = clahe.apply(lab_channels[0])
#     img_lab = cv.merge(lab_channels)
#     img_clahe = cv.cvtColor(img_lab, cv.COLOR_LAB2BGR)

#     return img_clahe


# # Remove (as much as possible) non-whites from the image
# def keepWhites(img, sensitivity):
#     img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
#     lower_white = np.array([0, 0, 255-sensitivity])
#     upper_white = np.array([180, sensitivity, 255])

#     mask = cv.inRange(img_hsv, lower_white, upper_white)
#     img_masked = cv.bitwise_and(img, img, mask=mask)

#     return img_hsv, img_masked

# for i in range(0, 26):
#     img = cv.imread('images/{}.jpg'.format(i))
#     img_clahe = applyCLAHE(img, 2.0, (8,8))
#     img_blurred = cv.blur(img_clahe, (21,1))
#     _, img_masked = keepWhites(img_clahe, 100)
    
#     cv.imshow('Blurred', img_blurred)
#     cv.imshow('Masked', img_masked)

#     cv.waitKey(0)

# cv.destroyAllWindows()