import SiSoPyInterface as s
import sys
import cv2
from matplotlib import pyplot as plt
from math import isclose
import numpy as np
import SDZ_v4 as SDZ

# Definitions #
###############################################################################
# returns count of available boards
def getNrOfBoards():
    nrOfBoards = 0
    (error, buffer, buflen) = s.Fg_getSystemInformation(None, s.INFO_NR_OF_BOARDS, s.PROP_ID_VALUE, 0)
    if error == s.FG_OK:
        nrOfBoards = int(buffer)
    return nrOfBoards
###############################################################################
# Lets the user select one of the available boards, returns the selected board, or -1 if nothing is selected
def selectBoardDialog():
    maxNrOfboards = 10
    nrOfBoardsFound = 0
    nrOfBoardsPresent = getNrOfBoards()
    maxBoardIndex = -1
    minBoardIndex = None

    if nrOfBoardsPresent <= 0:
        print("No Boards found!")
        return -1

    print('Found', nrOfBoardsPresent, 'Board(s)')

    for i in range(0, maxNrOfboards):
        skipIndex = False
        boardType = s.Fg_getBoardType(i)
        if boardType == s.PN_MICROENABLE5_LIGHTBRIDGE_VCL:
            boardName = "MicroEnable 5 LB-VCL"
        elif boardType == s.PN_MICROENABLE5_MARATHON_VCLx:
            boardName = "MicroEnable 5 MA-VCLx"
        elif boardType == s.PN_MICROENABLE5_MARATHON_VCL:
            boardName = "MicroEnable 5 MA-VCL"
        else:
            boardName = "Unknown / Unsupported Board"
            skipIndex = True

        if not skipIndex:
            sys.stdout.write("Board ID " + str(i) + ": " + boardName + " 0x" + format(boardType, '02X') + "\n")
            nrOfBoardsFound = nrOfBoardsFound + 1
            maxBoardIndex = i
            if minBoardIndex is None: minBoardIndex = i

        if nrOfBoardsFound >= nrOfBoardsPresent:
            break

        if nrOfBoardsFound < 0:
            break
    if nrOfBoardsFound <= 0:
        print("No Boards found!")
        return -1
    inStr = "=====================================\n\nPlease choose a board[{0}-{1}]: ".format(minBoardIndex, maxBoardIndex)
#	userInput = input(inStr)
    userInput = 0

    return int(userInput)

################################################################################
# Takes in a open cv image or image array and computes the centroids of the image
# Output is array of X, Y positions of the found centroids
def initAndCentroid(Image):
    #Binirize
    ret, Binary = cv2.threshold(Image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Find contours to then calculate the centroids
    contours, hierarchy = cv2.findContours(Binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #
    #
    Centroids = np.zeros((1, 2))
    for c in contours:
        Mts = cv2.moments(c)
        if Mts["m00"] != 0:
            cX = (Mts["m10"] / Mts["m00"]) # Moment for Ref in X corresponding to contour
            cY = (Mts["m01"] / Mts["m00"]) # Moment for Ref in Y
        else:
            cX, cY = 0, 0
        Centroids = np.append(Centroids, [[cX, cY]], axis=0) #Creating list of centroids for Ref

    indices = [i for i, x in enumerate(Centroids) if x[0] == 0 and x[1] == 0 ]
    for index in sorted(indices, reverse=True):     #Step backwards so you dont mess up the index order
        Centroids = np.delete(Centroids, index, axis=0)

    np.savetxt('RefCentroids.txt', Centroids)
    return Centroids
################################################################################
# Uses Centroid locations to find lines to calculate the lenslet array alignment tilt
# returns averaged angle for rolling averaged slope
def findAngle(coordArray, xStart, length):
    xVal = coordArray[xStart, 0]
    yVal =coordArray[xStart, 1]
    match = 0
    i = 0
    STOP = False
    angle = np.empty(0)
    ptsCoord = np.array([[xVal, yVal]])
    while match < length:
        while i < len(coordArray) and not STOP:
            if isclose(xVal, coordArray[i, 0], abs_tol=30) and isclose(yVal,coordArray[i, 1], abs_tol=10):
                    xVal, yVal = coordArray[i, :]
                    ptsCoord = np.append(ptsCoord, [[xVal, yVal]], axis=0)
                    coordArray = np.delete(coordArray, i, axis=0)
                    match += 1
                    STOP = True
            else: i += 1
            if i >= len(coordArray):
                break
        if i >= len(coordArray):
            break
        STOP = False
        i = 0
    for j in range(len(ptsCoord)-1):
        if (ptsCoord[j + 1, 0] - ptsCoord[j, 0]) == 0:
            slopeM = 0
        else:
            slopeM = (ptsCoord[j + 1, 1] - ptsCoord[j, 1]) / (ptsCoord[j + 1, 0] - ptsCoord[j, 0])
        angle = np.append(angle, -np.arctan(slopeM) * (180 / np.pi))
    return np.mean(angle), ptsCoord

################################################################################
# MAIN #
# Board and applet selection #
boardId = selectBoardDialog()

if boardId < 0:
    exit(1)

# definition of resolution
width = 2560
height = 2048
samplePerPixel = 1
bytePerSample = 1
isSlave = False
useCameraSimulator = True
camPort = s.PORT_A

# number of buffers for acquisition
nbBuffers = 20
totalBufferSize = width * height * samplePerPixel * bytePerSample * nbBuffers

# number of image to acquire
nrOfPicturesToGrab = 1
frameRate = 1

boardType = s.Fg_getBoardType(boardId)
#applet = "JaiGoBasic"
(err, applet) = s.Fg_findApplet(boardId)
if err != 0:
    print("No applet is found")
    exit(0)
else:
    print('Applet found:', applet)
# INIT FRAMEGRABBER
print('Initializing Board ..', end='')

if isSlave:
    fg = s.Fg_InitEx(applet, boardId, 1)
else:
    fg = s.Fg_InitEx(applet, boardId, 0)

# error handling
err = s.Fg_getLastErrorNumber(fg)
mes = s.Fg_getErrorDescription(err)

if err < 0:
    print("Error", err, ":", mes)
    sys.exit()
else:
    print("ok")

# allocating memory
memHandle = s.Fg_AllocMemEx(fg, totalBufferSize, nbBuffers)

# Set Applet Parameters
err = s.Fg_setParameterWithInt(fg, s.FG_WIDTH, width, camPort)
if err < 0:
    print("Fg_setParameter(FG_WIDTH) failed: ", s.Fg_getLastErrorDescription(fg))
    s.Fg_FreeMemEx(fg, memHandle)
    s.Fg_FreeGrabber(fg)
    exit(err)

err = s.Fg_setParameterWithInt(fg, s.FG_HEIGHT, height, camPort)
if err < 0:
    print("Fg_setParameter(FG_HEIGHT) failed: ", s.Fg_getLastErrorDescription(fg))
    s.Fg_FreeMemEx(fg, memHandle)
    s.Fg_FreeGrabber(fg)
    exit(err)
if useCameraSimulator:
    # Start Generator
    s.Fg_setParameterWithInt(fg, s.FG_GEN_ENABLE, s.FG_GENERATOR, camPort)
else:
    s.Fg_setParameterWithInt(fg, s.FG_GEN_ENABLE, s.FG_CAMPORT, camPort)
# Read back settings
(err, oWidth) = s.Fg_getParameterWithInt(fg, s.FG_WIDTH, camPort)
if err == 0:
    print('Width =', oWidth)
(err, oHeight) = s.Fg_getParameterWithInt(fg, s.FG_HEIGHT, camPort)
if err == 0:
    print('Height =', oHeight)
(err, oString) = s.Fg_getParameterWithString(fg, s.FG_HAP_FILE, camPort)
if err == 0:
    print('Hap File =', oString)
# Create Display window dummy for error loops
dispId0 = 0
err = s.Fg_AcquireEx(fg, camPort, nrOfPicturesToGrab, s.ACQ_STANDARD, memHandle)
if err != 0:
    print('Fg_AcquireEx() failed:', s.Fg_getLastErrorDescription(fg))
    s.Fg_FreeMemEx(fg, memHandle)
    s.Fg_FreeGrabber(fg)
    exit(err)

last_pic_nr = 0
img = "will point to last grabbed image"
nImg = "will point to Numpy image/matrix"
win_name_img = "Source Image (SiSo Runtime)"
win_name_res = "Result Image (openCV)"
STOP_LOOPING = 0
P = {
    "f" : 4100,           # Focal Length in units [um]
    "pixel Size" : 5.0,    # Pixel Size in units [um]
    "Pixels Per Lens" : 30,
    "Pixels" : 1750, # per dimension
    "Nodes" : 59, # number of lenses
}
linePts = np.array([[0, 0]])
startList = [500, 1000, 1500, 2000, 750]
theta = np.zeros(len(startList))
thetaIter = 0
print("Acquisition started")

cur_pic_nr = s.Fg_getLastPicNumberBlockingEx(fg, last_pic_nr + 1, camPort, 5, memHandle)
if cur_pic_nr < 0:
    print("Fg_getLastPicNumberBlockingEx(", (last_pic_nr + 1), ") failed: ", (s.Fg_getLastErrorDescription(fg)))
    s.Fg_stopAcquire(fg, camPort)
    s.Fg_FreeMemEx(fg, memHandle)
    s.Fg_FreeGrabber(fg)
    exit(cur_pic_nr)

last_pic_nr = cur_pic_nr
# Get image
img = s.Fg_getImagePtrEx(fg, last_pic_nr, camPort, memHandle)
nImg = s.getArrayFrom(img, width, height)
thetaIter = 0
# Centroid Calculations
centArray = initAndCentroid(nImg)
minCol = np.min(centArray[:,0])
maxCol = np.max(centArray[:,0])
minRow = np.min(centArray[:,1])
maxRow = np.max(centArray[:,1])

if maxCol - minCol > maxRow - minRow:
    pxSpan = maxCol - minCol
else:
    pxSpan = maxRow - minRow
P["Pixels"] = pxSpan

for s in startList:
    theta[thetaIter], ptsArray = findAngle(centArray, s, 20)
    linePts = np.append(linePts, ptsArray, axis=0)
    thetaIter += 1
avgTheta = np.mean(theta)

print ("Theta angle: %.4f \n" % avgTheta)

(h, w) = nImg.shape[:2]
center = (w / 2, h / 2)

M = cv2.getRotationMatrix2D(center, -avgTheta, 1)
rotated = cv2.warpAffine(nImg, M, (h, w))
buff = 50
rotated = rotated[int(minRow - buff):int(maxRow + buff), int(minCol - buff):]
# Doing the Zonal Reconstruct Algo + Plotting
RefFile = 'RefCentroids.txt'
W, Sx, Sy = SDZ.computeZR(P, RefFile, rotated)
SDZ.plotZR(W)
SDZ.saveZR(W)
# Plotting the lines used for rotation calculation
fig2, ax = plt.subplots(figsize=(15,9))
ax.imshow(nImg, cmap=plt.cm.gray)

for q in range(len(linePts)):
    ax.plot(linePts[q][0], linePts[q][1], 'or', markersize=5)  # reference
ax.text(1000, 100, 'Theta '+str(round(avgTheta,5)),fontsize=20, color='red',horizontalalignment='center')
plt.show()
# Display Image post-rotation
fig, ax = plt.subplots(figsize=(15,9))
plt.imshow(rotated, cmap='gray')
plt.show()

print("Acquisition stopped")
# Clean up
print("Exited.")