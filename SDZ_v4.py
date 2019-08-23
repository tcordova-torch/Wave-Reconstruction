# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 07:20:47 2019
Last Modified July 5 2019
"""
import numpy as np
import cv2
import matplotlib
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from zonalReconstruct import zonalReconstruction


plt.close('all')
###############################################################################
# Import, Convert to Grayscale, and threshold binary of Reference
def importAndFindCentroids(P, refFile, devImage):
    # Import, Convert to Grayscale, and threshold binary of Deviated
    ret, DevBinary = cv2.threshold(devImage,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Find contours to then calculate the centroids
    contours2, hierarchy2 = cv2.findContours(DevBinary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    RefCentroids = []
    DevCentroids = []

    try: # Checking if Reference centroids already calculated
        RefCentroids = np.loadtxt('RefCentroids.txt')
    except RefCentroids is None:
        # In event that reference does not exist, do computations
        RefImage = cv2.cvtColor(refFile, cv2.COLOR_RGB2GRAY)
        ret, RefBinary = cv2.threshold(RefImage,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours1, hierarchy1 = cv2.findContours(RefBinary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours1:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = (M["m10"] / M["m00"])  # Moment for Ref in X corresponding to contour
                cY = (M["m01"] / M["m00"])  # Moment for Ref in Y
            else:
                cX, cY = 0, 0
            RefCentroids.append((cX, cY))  # Creating list of centroids for Ref
        np.savetxt('RefCentroids.txt', RefCentroids)
    # Calculating contours for dev Image
    for c in contours2:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = (M["m10"] / M["m00"])
            cY = (M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        DevCentroids.append((cX, cY)) # Creating list of centroids for Dev

    T = DevCentroids  # Temp Array for manipulation
    # This method needs work because what happens when a spot is skipped or does not show up
    # It assumes the place in the array makes it a neighbor
    RefCoord = []  # Array for x,y paired ref centroid positions
    DevCoord = []  # Array for corresponding dev centroid positions

    i = 0; j = 0  # iterating Parameters
    STOP = 0; match = 0 # counting and condition parameters
    delta = 17.0  # arbitrary tolerance for defining a match
    x2 = 0; y2 = 0
    while i < len(RefCentroids):
        x1 = RefCentroids[i][0]
        y1 = RefCentroids[i][1]
        while j < len(T) and STOP == 0:
            if x1 - delta <= T[j][0] <= x1 + delta and y1 - delta <= T[j][1] <= y1 + delta:
                x2 = T[j][0]
                y2 = T[j][1]
                np.delete(T, j, 0)
                match += 1
                STOP = 1
            else:  # if no match Ref pairs with itself, could be replaced
                x2 = x1
                y2 = y1
                j += 1
        STOP = 0
        j = 0

        RefCoord.append((x1, y1))
        DevCoord.append((x2, y2))
        i += 1

    RefCoord = np.array(RefCoord)
    DevCoord = np.array(DevCoord)
    Ref, Dev = Coord2Matrix(P, RefCoord, DevCoord)
    Sx, Sy = scaling(P, Dev, Ref)
    return Sx, Sy

###############################################################################
def Coord2Matrix(P, RefCoord, DevCoord):
    nbPixels = P["Pixels"] # per dimension
    nbNodes = P["Nodes"]
    # Bin array discretizes the Ref x,y positions to convert coordinate lists to matrices
    binCoord = np.zeros((len(RefCoord),2))  # Binned coordinates, i.e. discretized
    for i in range(len(DevCoord)):
        binCoord[i,0] = nbNodes * RefCoord[i, 0] / nbPixels # discrete X positions
        binCoord[i,1] = nbNodes * RefCoord[i,1] / nbPixels # discrete Y positions

    maxRow = int(np.max(binCoord[:,0]))
    minRow = int(np.min(binCoord[:,0]))
    maxCol = int(np.max(binCoord[:,1]))
    minCol = int(np.min(binCoord[:,1]))
    if maxCol > maxRow: nbNodes = maxCol+1
    else: nbNodes = maxRow+1

    dX = np.zeros((nbNodes, nbNodes))  # Discretized X position matrix
    dY = np.zeros((nbNodes, nbNodes))  # Discretized Y position matrix
    # Creating the dX and dY matrices from the difference in centroids positions at
    # the corresponding bin location for X and Y respectively
    for j in range(len(binCoord)):
        m = int(binCoord[j, 0])
        n = int(binCoord[j, 1])
        dX[m, n] = DevCoord[j, 0] #- RefCoord[j, 0]
        dY[m, n] = DevCoord[j, 1] #- RefCoord[j, 1]

    dX = dX[minRow:maxRow, minCol:maxCol]
    dY = dY[minRow:maxRow, minCol:maxCol]
    return dX, dY

###############################################################################
def scaling(P, dX, dY):
         # nb lenses in x/y-direction
         Sx = dX * P["pixel Size"] / P["f"] * P["Pixels Per Lens"] * P["pixel Size"]
         Sy = dY * P["pixel Size"] / P["f"] * P["Pixels Per Lens"] * P["pixel Size"]
         return Sx, Sy

###############################################################################
def plotZR(W):
    # creating X and Y matrices to use as coordinates for 3D plot
    n = len(W)
    wX = np.zeros((n, n))
    wY = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            wX[i, j] = i
            wY[j, i] = i

    minima = np.min(np.min(W))
    if minima < 0:
        minima = abs(minima)
    else: minima = 0
    # 3D Plotting
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(wX, wY, W + minima, cmap=cm.plasma)
    fig.colorbar(surf, shrink=0.5)
    plt.show()

###############################################################################
def computeZR(Parameters, refFile, devFile):
    Sx, Sy = importAndFindCentroids(Parameters, refFile, devFile)
    W = zonalReconstruction(Sx, Sy, 1)
    return W, Sx, Sy

###############################################################################
def saveZR(W):
    matplotlib.image.imsave('reconstructed.bmp', W)
    return 0

# END #