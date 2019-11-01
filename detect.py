import numpy as np
import cv2 as cv
import math
import argparse
from scipy.spatial import distance as dist

debug = False

def capture_frame(cap):
    stop = False
    captured = False
    ret, frame = cap.read()
    if ret == True:
        ret, frame = cap.read()
        cv.imshow('Video', frame)
        k = cv.waitKey(1)
        
        if k == 27:     # ESC pressed
            stop = True
        elif k == 13:   # ENTER pressed
            captured = True
    else:
        stop = True

    return stop, captured, frame

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

    if debug:
        cv.imshow('Localization: Original', img)
        cv.imshow('Localization: Original rotated', img_rotated)
        cv.imshow('Localization: CLAHE', img_clahe)
        cv.imshow('Localization: Gray', img_gray)
        cv.imshow('Localization: Threshold', img_thresh)
        cv.imshow('Localization: Countours Filter', img_filtered_contours)
        cv.imshow('Localization: Closed', img_closed)
        cv.imshow('Localization: Targets Filter', img_filtered_targets)

    return rois

# https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/ 
def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    return np.array([tl, tr, br, bl], dtype="float32")


def getPerspectiveBarcode(dilated, img):
    barSize = 400
    cnts, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    rect = cv.minAreaRect(cnts[0])
    pts = order_points(cv.boxPoints(rect))
    (tl, tr, br, bl) = pts

    bCut = (bl[1]-tl[1]) * 0.2     # cut 20% bottom (barcode numbers)
    
    pts1 = np.float32([tl, tr, [bl[0],bl[1]-bCut],  [br[0],br[1]-bCut]])
    pts2 = np.float32([[0,0],[barSize,0],[0,barSize], [barSize,barSize]])
    M = cv.getPerspectiveTransform(pts1,pts2)
    dst = cv.warpPerspective(img, M,(barSize,barSize))
    
    # get bars using threshold
    img_clahe = applyCLAHE(dst, 4.0, (8,8))
    img_gray = cv.cvtColor(img_clahe, cv.COLOR_BGR2GRAY)

    img_thresh = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize=41, C=10)
    
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 3))
    img_closed = cv.morphologyEx(img_thresh, cv.MORPH_CLOSE, kernel, iterations=1 ) # remove noise

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))
    img_closed = cv.morphologyEx(img_closed, cv.MORPH_OPEN, kernel, iterations=3)   # join bars vertically

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 5))
    img_closed = cv.morphologyEx(img_closed, cv.MORPH_CLOSE, kernel, iterations=1 ) # remove noise

    if debug:
        cv.imshow('Decoding: Perspective Barcodes', dst)
        cv.imshow('Decoding: Adaptive threshold', img_thresh)
        cv.imshow('Decoding: Adaptive threshold after close', img_closed)
        cv.imshow('Decoding: Adaptive threshold after open', img_closed)
        cv.imshow('Decoding: Adaptive threshold after close2', img_closed)

    return (dst, img_closed, M)


def scanLines(img, pos):
    lineImg = np.zeros(img.shape[:2], np.uint8)

    if pos == "middle":
        cv.line(lineImg, (0, 200), (400, 200), 255, 2, cv.LINE_AA)    
    elif pos == "top":
        cv.line(lineImg, (0, 100), (400, 100), 255, 2, cv.LINE_AA)

    lineImg[lineImg > 0] = 255  # put all values at 255 (to fix OpenCV's smoothing)
    
    bar_lines = cv.bitwise_and(img, img, mask=lineImg)

    bar_lines_whites = np.array(np.where(bar_lines == 255))
    first_white_pixel = bar_lines_whites[:, 0][1]
    last_white_pixel = bar_lines_whites[:, -1][1]

    blank_lines = lineImg.copy()
    blank_lines[bar_lines > 0] = 0
    blank_lines[:, :first_white_pixel] = 0
    blank_lines[:, last_white_pixel:] = 0

    # number of lines in the scan
    contours, _ = cv.findContours(bar_lines, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    nLines = len(contours)

    if debug:
        cv.imshow("Decoding: Scan line " + pos, lineImg)
        cv.imshow("Decoding: Barcode scanned " + pos, bar_lines)

    return (lineImg, bar_lines, blank_lines, nLines)



def scanBarcode(ori, barcode, barcodeThr, M):
    height, width = ori.shape[:2]
    barcodeThr = cv.bitwise_not(barcodeThr)     # invert barcode
    
    # select the scan line that gets more lines
    (scanLine, barScanned, blankScanned, nLinesM) = scanLines(barcodeThr, "middle")
    (scanLineTop, barScannedTop, blankScannedTop, nLinesT) = scanLines(barcodeThr, "top")
    if nLinesT > nLinesM:
        barScanned = barScannedTop
        blankScanned = blankScannedTop
        scanLine = scanLineTop

    barcode[barScanned > 0] = (0, 0, 255)
    barcode[blankScanned > 0] = (255, 0, 0)

    # barcode to original perspective
    cv.warpPerspective(barcode, M, (width, height), dst=ori,
                       borderMode=cv.BORDER_TRANSPARENT, flags=cv.WARP_INVERSE_MAP)

    if debug:
        cv.imshow('Decoding: Scan result', ori)
        cv.imshow("Decoding: Barcode inverted", barcodeThr)
        cv.imshow("Decoding: Barcode with scan", barcode)

    return (ori, barScanned, blankScanned)

 

def linePercentage(bars, blanks):
    # get line length by the dotted line
    # linesP = cv.HoughLinesP(bars, 1, np.pi / 180, 10, None, 50, 20)
    # (x1, y1, x2, y2) = linesP[0][0]
    # img = cv.cvtColor(bars, cv.COLOR_GRAY2BGR)
    # cv.line(img, (x1,y1), (x2,y2), (0, 255, 0), 2)
    # cv.imshow('Line Percentage Hough', img)
    # totalSize = math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
    # totalSize = x2 - x1 (vs above?)

    bar_lines_whites = np.array(np.where(bars == 255))
    first_white_pixel = bar_lines_whites[:, 0][1]
    last_white_pixel = bar_lines_whites[:, -1][1]
    totalSize = last_white_pixel - first_white_pixel

    linesInfo = []
    for color, img in [('B', bars), ('W', blanks)]:
        contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            rect = cv.minAreaRect(c)
            pts = cv.boxPoints(rect)
            pts = order_points(pts).astype(np.int32)
            (tl, tr, _, _) = pts

            # size = math.sqrt( ((tr[0]-tl[0])**2) + ((tr[1]-tl[1])**2) )
            size = tr[0] - tl[0] + 1
            linesInfo.append((tl, tr, color, size))

    linesInfo.sort(key = lambda x: x[0][0])

    acc = 0
    for tl, tr, color, size in linesInfo:
        perc = (size / totalSize) * 100
        acc += perc
        print("{0}: {1:.2f}".format(color, perc), end = '%  ', flush=True)
    print()
    print(acc)

    #print(linesInfo)

    # for img in [blanks, bars]:
    #     # get countours
    #     contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #     img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    #     acc = 0
    #     for c in contours:
    #         rect = cv.minAreaRect(c)
    #         pts = cv.boxPoints(rect)
    #         pts = order_points(pts).astype(np.int32)
    #         (tl, tr, _, _) = pts

    #         cv.drawContours(img, [pts], -1, (0, 0, 255), 1, cv.LINE_AA)
            
    #         size = math.sqrt( ((tr[0]-tl[0])**2) + ((tr[1]-tl[1])**2) )

    #         perc = (size / totalSize) * 100
    #         acc += perc
    #         print("%.2f" % perc, end = '%  ', flush=True)
    #     print()

    if debug:
        print("Total size: ", totalSize)
        print("N contours: ", len(contours))
        print("\nTotal percentage", acc)
        cv.imshow("Decoding: Scan line contours", img)

def decodeBar(mask, img):
    np.multiply(mask, 255)  
    
    # dilate barcode mask
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
    dilated = cv.dilate(mask, kernel, iterations=2)
    code_dilated = cv.bitwise_and(img, img, mask=dilated)

    (barcodeImg, barcodeThr, M) = getPerspectiveBarcode(dilated, img)

    (final, bars, blanks) = scanBarcode(img, barcodeImg, barcodeThr, M)
    
    if debug:
        cv.imshow('Decoding: Barcode Image', code_dilated)
    
    linePercentage(bars, blanks)

    return final
    


def showResult(img, rois):
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for roi in rois:
        mask = cv.bitwise_or(mask, roi)

    # cv.imshow('Barcodes', cv.bitwise_and(img, img, mask=mask))

    img_ = img.copy()

    # show in original image  
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, cnts, -1, (0,255,0), 1)
    cv.imshow('Localization: Detected Barcodes', img)

    final = decodeBar(mask, img_)

    cv.imshow('Decoding: Final', final)

    cv.waitKey(0)


for i in range(0, 27):
    img = cv.imread('images/{}.jpg'.format(i))
    barcodes = getBarcodes(img, 0)
    showResult(img, barcodes)

# parser = argparse.ArgumentParser(description="A program that detects barcodes in images")
# parser.add_argument('--scan', help="Scan direction", choices=('vertical', 'horizontal'), default='vertical', type=str, metavar='')
# parser.add_argument('--debug', action='store_true')
# exclusive_group = parser.add_mutually_exclusive_group(required=True)
# exclusive_group.add_argument('--image', help='Path to image being scanned', type=str)
# exclusive_group.add_argument('--video', help='Use computer connected camera to retrieve images', action='store_true')
# args = parser.parse_args()

# debug = args.debug
# angle = 0 if args.scan == 'vertical' else 90
# if args.video:
#     cap = cv.VideoCapture(0)
#     while (True):
#         stop, captured, frame = capture_frame(cap)
#         if stop:
#             break
#         elif captured:
#             barcodes = getBarcodes(frame, angle)
#             showResult(frame, barcodes)
# else:
#     img = cv.imread(args.image)
#     barcodes = getBarcodes(img, angle)
#     showResult(img, barcodes)
