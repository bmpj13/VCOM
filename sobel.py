import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Melhor aplicar thresholding depois do gradiente? (gradiente assim parece mais "denso")
# Testar: CLAHE

# MAUS: 6, 7, 8, 10, 11, 13, 14, 15, 16, 20, 22, 25
# Resolvidos: 8, 11, 16 (com thresholding antes do gradiente)

def calcSobelGradient(img):
    sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=-1)
    sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=-1)
    gradient = cv.subtract(sobelx, sobely)
    gradient = cv.convertScaleAbs(gradient)
    return gradient

img = cv.imread('images/6.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

(_, threshold) = cv.threshold(gray, thresh=190, maxval=255, type=cv.THRESH_BINARY)
#threshold = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 8)
cv.imshow('Threshold', threshold)

gradient = calcSobelGradient(threshold)
cv.imshow('Gradient', gradient)

bilateral = cv.bilateralFilter(gradient, 15, 50, 50)
cv.imshow('Bilateral', bilateral)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (27, 3))
closed = cv.morphologyEx(bilateral, cv.MORPH_CLOSE, kernel)
cv.imshow('Closed', closed)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
morphed = cv.erode(closed, kernel, iterations=4)
morphed = cv.dilate(morphed, kernel, iterations=4)
cv.imshow('Morphed', morphed)

contours, _ = cv.findContours(morphed.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
largestContour = max(contours, key=cv.contourArea)

x,y,w,h = cv.boundingRect(largestContour)
cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
cv.imshow('Contours', img)

#plt.hist(gray.ravel(),256,[0,256]); plt.show()

cv.waitKey(0)
cv.destroyAllWindows()