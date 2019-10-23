import numpy as np
import cv2 as cv
import argparse

harrisCornerThresh = 130    
houghThresh = 10    

src = cv.imread('images/20.jpg') 
src_gray = cv.cvtColor(src,cv.COLOR_BGR2GRAY)    

# Apply harris corner detection
    
blockSize = 3      # size of neighbourhood
apertureSize = 1   # aperture parameter of Sobel  
k = 0.04 

# detecting corners
dst = cv.cornerHarris(src_gray, blockSize, apertureSize, k)
        
# normalizing
dst_norm = np.empty(dst.shape, dtype=np.float32)
cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
dst_norm_scaled = cv.convertScaleAbs(dst_norm)
    
# image with corners in dots in white, black on the background
corners = np.copy(src_gray)
        
for i in range(dst_norm.shape[0]):
    for j in range(dst_norm.shape[1]):
        if int(dst_norm[i,j]) > harrisCornerThresh:
            corners[i,j] = 255
            cv.circle(dst_norm_scaled, (j,i), 1, (0), 2)
        else:
            corners[i,j] = 0 
            

# dilate for marking the corners 
element = cv.getStructuringElement(cv.MORPH_RECT, (2 , 2))
resultHough = cv.dilate(corners,element,iterations = 2)
resultHough = cv.erode(resultHough,element,iterations = 2)
    
cv.imshow('Corners detected', resultHough)
cv.imshow('Corners detected2', dst_norm_scaled)


# Apply hough transform

cdstP = np.copy(src)

minLinLength = 30
maxLineGap = 30
linesP = cv.HoughLinesP(resultHough, 1, np.pi / 180, houghThresh, None, minLinLength, maxLineGap)  
        
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
        
cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
cv.waitKey(0)
cv.destroyAllWindows()
    

 
