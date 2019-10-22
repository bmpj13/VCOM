import numpy as np
import cv2 as cv
from collections import Counter
import random

THETA_MIN_NUMBER_LINES = 20     # To be refined
MAX_DISTANCE_BETWEEN_LINES = 50  # To be refined


def plotHoughLines(img, lines, color=(0, 0, 255)):
    print(len(lines))
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 50*(-b))
        y1 = int(y0 + 50*(a))
        x2 = int(x0 - 50*(-b))
        y2 = int(y0 - 50*(a))
        cv.line(img, (x1, y1), (x2, y2), color, 2)
        cv.circle(img, (x0, y0), 5, color, -1)


def plotHoughLinesByRegion(img, regions):
    for lines in regions:
        color = (random.randint(0, 255), random.randrange(
            0, 255), random.randrange(0, 255))
        plotHoughLines(img, lines, color)


def splitLinesByRegion(lines):
    lines_abs = np.abs(np.array(lines))
    diff = lines_abs[1:, 0] - lines_abs[:-1, 0]
    split_indexes = np.where(diff > MAX_DISTANCE_BETWEEN_LINES)[0] + 1
    splits = np.split(lines, split_indexes)
    return splits


def mergeThetas(thetas, angleRange):
    merged_thetas = dict()
    for theta in thetas:
        found = False
        for angle in merged_thetas.keys():
            if angle-5 <= theta and theta <= angle+5:
                found = True
                break
        if found:
            merged_thetas[angle].append(theta)
        else:
            merged_thetas[theta] = [theta]
    return merged_thetas

img = cv.imread('images/0.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

meanGray = cv.mean(gray)[0]
edges = cv.Canny(gray, int(0.33 * meanGray), int(1.33 * meanGray), apertureSize=3, L2gradient=True)

cv.imshow('Gray', gray)
cv.imshow('Canny', edges)

hough_lines = cv.HoughLines(edges, 1, np.pi/180, 50)
# Extract the first (rho, theta) pair for each line (the first pair has the strongest confidence)
lines = [(line[0][0], line[0][1]) for line in hough_lines]
theta_counter = Counter([theta for (_, theta) in lines]).most_common()
thetas = [theta for (theta, count) in theta_counter if count > THETA_MIN_NUMBER_LINES]

# print(theta_counter)
# print(thetas)

merged_thetas = mergeThetas(thetas, 5)

for thetas in merged_thetas.values():
    linesWithTheta = [(rho,theta) for (rho,theta) in lines if theta in thetas]
    linesWithTheta = sorted(linesWithTheta, key=lambda tup: abs(tup[0]))
    img_copy = img.copy()
    splits = splitLinesByRegion(linesWithTheta)
    plotHoughLinesByRegion(img_copy, splits)
    cv.imshow('Hough Lines {0}'.format(thetas[0]), img_copy)
    # minRho = linesWithTheta[0][0]
    # maxRho = linesWithTheta[-1][0]


cv.waitKey(0)
cv.destroyAllWindows()
