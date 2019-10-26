import numpy as np
import cv2 as cv

# "CLAHE -> Filter Whites" melhor que "Original -> Filter Whites"

# Apply CLAHE to colored image by converting to LAB colorspace
def applyCLAHE(img, clipLimit, tileGridSize):
    img_lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    clahe = cv.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    lab_channels = cv.split(img_lab)
    lab_channels[0] = clahe.apply(lab_channels[0])
    img_lab = cv.merge(lab_channels)
    img_clahe = cv.cvtColor(img_lab, cv.COLOR_LAB2BGR)

    return img_clahe

def calcSobelGradient(img):
    sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=-1)
    sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=-1)
    gradient = cv.subtract(sobelx, sobely)
    gradient = cv.convertScaleAbs(gradient)
    return gradient


# Remove (as much as possible) non-grays from the image
# def keepGrays(img, satSensitivity):
#     img_copy = img.copy()
#     img_gray = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY) # usar o gray para manter brancos e pretos
#     img_hsv = cv.cvtColor(img_copy, cv.COLOR_BGR2HSV)

#     lower_black = np.array([0, 0, 0])
#     upper_black = np.array([180, 255, 50])
#     mask_black = cv.inRange(img_hsv, lower_black, upper_black)

#     lower_white = np.array([0, 0, 205])
#     upper_white = np.array([180, 0, 255])
#     mask_white = cv.inRange(img_hsv, lower_white, upper_white)

#     lower_colored = np.array([0, 0, 0])
#     upper_colored = np.array([180, satSensitivity, 255])
#     mask_colored = cv.inRange(img_hsv, lower_colored, upper_colored)

#     mask = cv.bitwise_or(mask_white, mask_black)
#     mask = cv.bitwise_or(mask, mask_colored)

#     mask[mask > 0] = 255
#     cv.imshow('Gray', img_gray)
#     cv.imshow('Mask', mask)

#     img_masked = cv.bitwise_and(img_copy, img_copy, mask=mask)

#     return img_hsv, img_masked


def keepGrays(img, tolerance):
    bg = cv.absdiff(img[:,:,0], img[:,:,1]) < tolerance
    gr = cv.absdiff(img[:,:,1], img[:,:,2]) < tolerance
    br = cv.absdiff(img[:,:,0], img[:,:,2]) < tolerance
    slices = np.bitwise_and(bg, gr, dtype=np.uint8)
    slices = np.bitwise_and(slices, br, dtype=np.uint8)

    img_masked = img.copy()
    img_masked[slices == 0] = [255, 255, 255]
    return img_masked

for i in range(0, 26):
    img = cv.imread('images/{}.jpg'.format(i))
    img_clahe = applyCLAHE(img, 5.0, (8,8))
    img_masked = keepGrays(img_clahe, 65)
    img_gray = cv.cvtColor(img_masked, cv.COLOR_BGR2GRAY)

    img_thresh1 = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 31, 30)
    
    cnts, _ = cv.findContours(img_thresh1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(cnts, key=cv.contourArea, reverse=True)

    thresh_copy = cv.cvtColor(img_thresh1, cv.COLOR_GRAY2BGR)
    for contour in contours:
        x,y,w,h = cv.boundingRect(contour)

        if h < 50:
            continue

        # if w*h < 500:
        #     break

        cv.rectangle(thresh_copy,(x,y),(x+w,y+h),(0,0,255),2)
        print(w,h, w*h)

    # img_blurred = cv.blur(img_thresh1, (15,1))
    # _, img_thresh2 = cv.threshold(img_blurred, 55, 255, cv.THRESH_BINARY)

    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 1))
    # closed = cv.morphologyEx(img_thresh2, cv.MORPH_CLOSE, kernel)

    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # morphed = cv.erode(closed, kernel, iterations=4)
    # morphed = cv.dilate(morphed, kernel, iterations=4)

    # cnts, _ = cv.findContours(morphed.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # contours = sorted(cnts, key=cv.contourArea, reverse=True)

    # for contour in contours:
    #     rbox = cv.minAreaRect(contour)
    #     pts = cv.boxPoints(rbox).astype(np.int32)
    #     cv.drawContours(img, [pts], -1, (0, 0, 255), 1, cv.LINE_AA)

    # cv.imshow('Original', img)
    cv.imshow('CLAHE', img_clahe)
    cv.imshow('Masked', img_masked)
    cv.imshow('Threshold 1', img_thresh1)
    cv.imshow('Rects Thresh 1', thresh_copy)
    # cv.imshow('Blurred', img_blurred)
    # cv.imshow('Threshold 2', img_thresh2)
    # cv.imshow('Closed', closed)
    # cv.imshow('Morphed', morphed)
    # cv.imshow('Image', img)

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







# for i in range(0, 26):
#     img = cv.imread('images/{}.jpg'.format(i))
#     img_clahe = applyCLAHE(img, 2.0, (8,8))
#     img_blurred = cv.blur(img_clahe, (21,1))
#     _, img_masked = keepWhites(img_clahe, 100)
    
#     cv.imshow('Blurred', img_blurred)
#     cv.imshow('Masked', img_masked)

#     cv.waitKey(0)

# cv.destroyAllWindows()