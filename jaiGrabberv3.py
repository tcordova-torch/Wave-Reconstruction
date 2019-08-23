#import subprocess
#subprocess.call(['/opt/SiliconSoftware/Runtime5.7.0/siso-rt5-5.7.0.76321-linux-amd64/setup-siso-env.sh'])

# IMPORT
print('Importing SiSo Wrapper')

import SiSoPyInterface as s

print('Runtime Version', s.Fg_getSWVersion())

# IMPORT additional modules
import sys
import cv2
from matplotlib import pyplot as plt
from scipy import stats

# for "s.getArrayFrom", to handle grabbed image as NumPy array
print('Importing NumPy', end='')
import numpy as np
print('Version', np.__version__)

# DEFINITIONS

# returns count of available boards
def getNrOfBoards():
    nrOfBoards = 0
    (err, buffer, buflen) = s.Fg_getSystemInformation(None, s.INFO_NR_OF_BOARDS, s.PROP_ID_VALUE, 0)
    if (err == s.FG_OK):
        nrOfBoards = int(buffer)
    return nrOfBoards

# Lets the user select one of the available boards, returns the selected board, or -1 if nothing is selected
def selectBoardDialog():
    maxNrOfboards = 10
    nrOfBoardsFound = 0
    nrOfBoardsPresent = getNrOfBoards()
    maxBoardIndex = -1
    minBoardIndex = None

    if (nrOfBoardsPresent <= 0):
        print("No Boards found!")
        return -1

    print('Found', nrOfBoardsPresent, 'Board(s)')

    for i in range(0, maxNrOfboards):
        skipIndex = False
        boardType = s.Fg_getBoardType(i)
        if boardType == s.PN_MICROENABLE4AS1CL:
            boardName = "MicroEnable IV AS1-CL"
        elif boardType == s.PN_MICROENABLE4AD1CL:
            boardName = "MicroEnable IV AD1-CL"
        elif boardType == s.PN_MICROENABLE4VD1CL:
            boardName = "MicroEnable IV VD1-CL"
        elif boardType == s.PN_MICROENABLE4AD4CL:
            boardName = "MicroEnable IV AD4-CL"
        elif boardType == s.PN_MICROENABLE4VD4CL:
            boardName = "MicroEnable IV VD4-CL"
        elif boardType == s.PN_MICROENABLE4AQ4GE:
            boardName = "MicroEnable IV AQ4-GE"
        elif boardType == s.PN_MICROENABLE4VQ4GE:
            boardName = "MicroEnable IV VQ4-GE"
        elif boardType == s.PN_MICROENABLE5AQ8CXP6B:
            boardName = "MicroEnable V AQ8-CXP"
        elif boardType == s.PN_MICROENABLE5VQ8CXP6B:
            boardName = "MicroEnable V VQ8-CXP"
        elif boardType == s.PN_MICROENABLE5VD8CL:
            boardName = "MicroEnable 5 VD8-CL"
        elif boardType == s.PN_MICROENABLE5AD8CL:
            boardName = "MicroEnable 5 AD8-CL"
        elif boardType == s.PN_MICROENABLE5AQ8CXP6D:
            boardName = "MicroEnable 5 AQ8-CXP6D"
        elif boardType == s.PN_MICROENABLE5VQ8CXP6D:
            boardName = "MicroEnable 5 VQ8-CXP6D"
        elif boardType == s.PN_MICROENABLE5AD8CLHSF2:
            boardName = "MicroEnable 5 AD8-CLHS-F2"
        elif boardType == s.PN_MICROENABLE5_LIGHTBRIDGE_ACL:
            boardName = "MicroEnable 5 LB-ACL"
        elif boardType == s.PN_MICROENABLE5_LIGHTBRIDGE_VCL:
            boardName = "MicroEnable 5 LB-VCL"
        elif boardType == s.PN_MICROENABLE5_MARATHON_VCLx:
            boardName = "MicroEnable 5 MA-VCLx"
        elif boardType == s.PN_MICROENABLE5_MARATHON_ACL:
            boardName = "MicroEnable 5 MA-ACL"
        elif boardType == s.PN_MICROENABLE5_MARATHON_ACX_SP:
            boardName = "MicroEnable 5 MA-ACX-SP"
        elif boardType == s.PN_MICROENABLE5_MARATHON_ACX_DP:
            boardName = "MicroEnable 5 MA-ACX-DP"
        elif boardType == s.PN_MICROENABLE5_MARATHON_ACX_QP:
            boardName = "MicroEnable 5 MA-ACX-QP"
        elif boardType == s.PN_MICROENABLE5_MARATHON_AF2_DP:
            boardName = "MicroEnable 5 MA-AF2-DP"
        elif boardType == s.PN_MICROENABLE5_MARATHON_VCL:
            boardName = "MicroEnable 5 MA-VCL"
        elif boardType == s.PN_MICROENABLE5_MARATHON_VCX_QP:
            boardName = "MicroEnable 5 MA-VCX-QP"
        elif boardType == s.PN_MICROENABLE5_MARATHON_VF2_DP:
            boardName = "MicroEnable 5 MA-VF2-DP"
        else:
            boardName = "Unknown / Unsupported Board"
            skipIndex = True

        if not skipIndex:
            sys.stdout.write("Board ID " + str(i) + ": " + boardName + " 0x" + format(boardType, '02X') + "\n")
            nrOfBoardsFound = nrOfBoardsFound + 1
            maxBoardIndex = i
            if minBoardIndex == None: minBoardIndex = i

        if nrOfBoardsFound >= nrOfBoardsPresent:
            break

        if nrOfBoardsFound < 0:
            break

    if nrOfBoardsFound <= 0:
        print("No Boards found!")
        return -1

    inStr = "=====================================\n\nPlease choose a board[{0}-{1}]: ".format(minBoardIndex, maxBoardIndex)
#	userInput = input(inStr)
    userInput=0

#	while (not userInput.isdigit()) or (int(userInput) > maxBoardIndex):
#		inStr = "Invalid selection, retry[{0}-{1}]: ".format(minBoardIndex, maxBoardIndex)
#		userInput = input(inStr)

    return int(userInput)

# MAIN

# Board and applet selection
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
nrOfPicturesToGrab = 50
frameRate = 60

# Get Loaded Applet
#boardType = s.Fg_getBoardType(boardId)
#if boardType == s.PN_MICROENABLE4AS1CL:
#	applet = "SingleAreaGray16"
#elif boardType == s.PN_MICROENABLE4AD1CL or boardType == s.PN_MICROENABLE4AD4CL or boardType == s.PN_MICROENABLE4VD1CL or boardType == s.PN_MICROENABLE4VD4CL:
#	applet = "DualAreaGray16"
#elif boardType == s.PN_MICROENABLE4AQ4GE or boardType == s.PN_MICROENABLE4VQ4GE:
#	applet = "QuadAreaGray16";
#else:
#	(err, applet) = s.Fg_findApplet(boardId)
#	if err != 0:
#		print("No applet is found")
#		exit(0)
#	else:
#		print('Applet found:', applet)

boardType = s.Fg_getBoardType(boardId)
applet = "JaiGoBasic"
(err, applet) = s.Fg_findApplet(boardId)
if err != 0:
    print("No applet is found")
    exit(0)
else:
    print('Applet found:', applet)



# INIT FRAMEGRABBER

print('Initializing Board ..', end='')

if isSlave:
    fg = s.Fg_InitEx(applet, boardId, 1);
else:
    fg = s.Fg_InitEx(applet, boardId, 0);

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
if (err < 0):
    print("Fg_setParameter(FG_WIDTH) failed: ", s.Fg_getLastErrorDescription(fg))
    s.Fg_FreeMemEx(fg, memHandle)
    s.Fg_FreeGrabber(fg)
    exit(err)

err = s.Fg_setParameterWithInt(fg, s.FG_HEIGHT, height, camPort)
if (err < 0):
    print("Fg_setParameter(FG_HEIGHT) failed: ", s.Fg_getLastErrorDescription(fg))
    s.Fg_FreeMemEx(fg, memHandle)
    s.Fg_FreeGrabber(fg)
    exit(err)

print(s.FG_BITALIGNMENT)
print(s.FG_LEFT_ALIGNED)
print(camPort)

#err = s.Fg_setParameterWithInt(fg, s.FG_BITALIGNMENT, s.FG_LEFT_ALIGNED, camPort)
#if (err < 0):
#	print("Fg_setParameter(FG_BITALIGNMENT) failed: ", s.Fg_getLastErrorDescription(fg))
#	s.Fg_FreeMemEx(fg, memHandle)
#	s.Fg_FreeGrabber(fg)
#	exit(err)

if useCameraSimulator:
    # Start Generator
    s.Fg_setParameterWithInt(fg, s.FG_GEN_ENABLE, s.FG_GENERATOR, camPort)
#	s.Fg_setParameterWithInt(fg, s.FG_GEN_ROLL, 1, camPort)
else:
    s.Fg_setParameterWithInt(fg, s.FG_GEN_ENABLE, s.FG_CAMPORT, camPort)

# Read back settings
(err, oWidth) = s.Fg_getParameterWithInt(fg, s.FG_WIDTH, camPort)
if (err == 0):
    print('Width =', oWidth)
(err, oHeight) = s.Fg_getParameterWithInt(fg, s.FG_HEIGHT, camPort)
if (err == 0):
    print('Height =', oHeight)
(err, oString) = s.Fg_getParameterWithString(fg, s.FG_HAP_FILE, camPort)
if (err == 0):
    print('Hap File =', oString)

# create a display window
#dispId0 = s.CreateDisplay(8 * bytePerSample * samplePerPixel, width, height)
#s.SetBufferWidth(dispId0, width, height)

# start acquisition
err = s.Fg_AcquireEx(fg, camPort, nrOfPicturesToGrab, s.ACQ_STANDARD, memHandle)
if (err != 0):
    print('Fg_AcquireEx() failed:', s.Fg_getLastErrorDescription(fg))
    s.Fg_FreeMemEx(fg, memHandle)
    s.CloseDisplay(dispId0)
    s.Fg_FreeGrabber(fg)
    exit(err)


cur_pic_nr = 0
last_pic_nr = 0
img = "will point to last grabbed image"
nImg = "will point to Numpy image/matrix"

win_name_img = "Source Image (SiSo Runtime)"
win_name_res = "Result Image (openCV)"

print("Acquisition started")

# RUN PROCESSING LOOP for nrOfPicturesToGrab images
#end_grab=0;
#while end_grab == 0:
STOP = 0

while STOP != 1:
    cur_pic_nr = s.Fg_getLastPicNumberBlockingEx(fg, last_pic_nr + 1, camPort, 5, memHandle)

    #if (cur_pic_nr < 0):
     #   print("Fg_getLastPicNumberBlockingEx(", (last_pic_nr + 1), ") failed: ", (s.Fg_getLastErrorDescription(fg)))
      #  s.Fg_stopAcquire(fg, camPort)
       # s.Fg_FreeMemEx(fg, memHandle)
       # s.CloseDisplay(dispId0)
       # s.Fg_FreeGrabber(fg)
       # exit(cur_pic_nr)

    last_pic_nr = cur_pic_nr

    # get image pointer
    img = s.Fg_getImagePtrEx(fg, last_pic_nr, camPort, memHandle)
    #    # handle this as Numpy array (using same memory, NO copy)
    nImg = s.getArrayFrom(img, width, height)
    #AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAaaaaAAAAAAAAAAAAAAAAAAA**********************************#
    #**********************************************************************************************************************#
    Image = nImg #cv2.cvtColor(nImg, cv2.COLOR_RGB2GRAY)

    ret, Binary = cv2.threshold(Image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Find contours to then calculate the centroids
    contours, hierarchy = cv2.findContours(Binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #
    #
    Centroids = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = (M["m10"] / M["m00"]) # Moment for Ref in X corresponding to contour
            cY = (M["m01"] / M["m00"]) # Moment for Ref in Y
        else:
            cX, cY = 0, 0
        Centroids.append((cX, cY)) #Creating list of centroids for Ref

    STOP = 0; match = 0; pMatch = 0; maxx = np.zeros(25); maxy = 0  # counting and condition parameters
    delta = 20.0  # arbitrary tolerance for defining a match


    fig, ax = plt.subplots(figsize=(15,9))
    ax.imshow(Image, cmap=plt.cm.gray)

    #******************************************************************************#
    #******************************************************************************#
    #Starting Positions
    #This section calculates the SubDivCt=4 starting positions
    #The starting positions all share the same x value located on the left side of the
    #spot array, approximatly 25% to the righ of the left most spot
    #They are then distributed evenly in the vertical
    #To do this, I am finding the top, botton, left, right of the spotfield to get
    #the diameter and center of the spotfield and using that to find the evenly
    #distributed starting positions starting and ending at 10% of the circle edge

    #im getting a few (0,0) centroid locations which is incorrect. This is to remove them
    indices = [i for i, x in enumerate(Centroids) if x == (0,0)]
    for index in sorted(indices, reverse=True):     #Step backwards so you dont mess up the index order
        Centroids.pop(index)

    DevCoordArray= np.array(Centroids)

    leftX = min(DevCoordArray[:,0])
    rightX = max(DevCoordArray[:,0])
    topY = min(DevCoordArray[:,1])
    bottomY = max(DevCoordArray[:,1])

    #leftY = topY+(bottomY-topY)/2
    #rightY = leftY
    #topX = leftX+(rightX-leftX)/2
    #bottomX = topX

    SubDivCt = 6        #Number of circle (spot field diameter) subdivisions

    startX = (rightX-leftX)*.25+leftX

    lnstart = topY+(bottomY-topY)*(1/10)
    lnstop = bottomY-(bottomY-topY)*(1/10)
    startY = np.linspace(lnstart,lnstop, SubDivCt )
    startY= np.array(startY)

    #for x in np.nditer(startY):
    #    ax.plot(startX, x, 'oy', markersize=5)

    #ax.plot(leftX, leftY, 'ob', markersize=5)
    #ax.plot(rightX, rightY, 'ob', markersize=5)
    #ax.plot(topX, topY, 'ob', markersize=5)
    #ax.plot(bottomX, bottomY, 'ob', markersize=5)

    #******************************************************************************#
    #******************************************************************************#

    theta=[]       #Used to store calculated theta values ater each iteration

    count = 0
    for starty in np.nditer(startY):
        startx = startX
        DevCoord = [] # Array for corresponding dev centroid positions
        x2 = 0
        y2 = 0
        XDEV = 0
        YDEV = 0
        i = 0; j = 0  # iterating Parameters
        while i < 30: #Find 30 centroids in a row for slope calculation

            x1 = startx + XDEV
            y1 = starty + YDEV
            while j < len(Centroids) and STOP == 0:
                #Nearest neighbor centroid finder with 'delta' as spatial threshold
                if x1 - delta <= Centroids[j][0] <= x1 + delta and y1 - delta <= Centroids[j][1] <= y1 + delta:
                    x2 = Centroids[j][0]
                    y2 = Centroids[j][1]
                    np.delete(Centroids, j, 0)
                    STOP = 1
                else:
                    j += 1
            STOP = 0
            j = 0
            startx,starty = x2,y2        #Centroid is new start position

            DevCoord.append((x2, y2))
            if i == 0:
                XDEV,YDEV = 30 , 0        #lenslet spacing is ~30px, 0 assumes lenslet is
            else:                         #basically aligned.
                DevCoordstep= np.array(DevCoord)
                X1 = DevCoordstep[:,0]        #Takes all currently calculated centroids
                Y1 = DevCoordstep[:,1]
                r = stats.linregress(X1, Y1)  #Uses linear regression to find slope from least squares fit
                XDEV = np.cos(r.slope)*30
                YDEV = np.sin(r.slope)*30
            i += 1

        DevCoord = np.array(DevCoord)
        X = []
        Y = []
        #for point in DevCoord:
        #    ImgCirc = cv2.circle(Image,DevCoord(point), 10, (0,0,255), -1)
        #cv2.imshow('Original Image',ImgCirc)
        k=0

        while k < len(DevCoord):
            x1o = DevCoord[k][0]
            y1o = DevCoord[k][1]
            ax.plot(x1o, y1o, 'or', markersize=2)         #reference
            k+=1

        X = DevCoord[:,0]
        Y = DevCoord[:,1]

        r= stats.linregress(X, Y)
        theta.append(np.arctan(r.slope)*(180/np.pi))



    avgTheta = np.mean(theta)
    print(avgTheta)
    ax.text(1000, 100, 'Theta '+str(round(avgTheta,5)),fontsize=20, color='red',horizontalalignment='center')
    #x=np.linspace(0,2000,10000)
    #y=slope*x+intercept
    #plt.plot(x, y, '-r')
    plt.show()



    #Resize
    #Centroid
    #Points in row
    #Calculate angle

    (h, w) = Image.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, avgTheta, 1)
    rotated = cv2.warpAffine(Image, M, (h, w))
    #cv2.imshow('Original Image',Image)
    cv2.imshow('Rotated Image',rotated)
    #cv2.namedWindow('image',WINDOW_NORMAL)
    cv2.resizeWindow('image', 600,600)
    cv2.waitKey(30)
    print("End = 1, Continue = else :\n")
    STOP = input()
    if STOP == 1:
        print("Acquisition stopped")

        # Clean up
        if (fg != None):
            s.Fg_stopAcquire(fg, camPort)
            s.Fg_FreeMemEx(fg, memHandle)
            s.Fg_FreeGrabber(fg)

        print("Exited.")
        break
    #    #nImg = cv2.medianBlur(nImg,3)
    #    th3 = cv2.adaptiveThreshold(nImg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    #    v = np.median(th3)
    #    sigma=0.3
    #    lower = int(max(0, (1.0 - sigma) * v))
    #    upper = int(min(255, (1.0 + sigma) * v))
    #    edges = cv2.Canny(nImg,upper,lower)
    #
    #    cImg = cv2.applyColorMap(nImg, cv2.COLORMAP_BONE)
    #    cv2.imshow('color',cImg)
        #cv2.imshow('frame',nImg)
    #    if cv2.waitKey(1) & 0xFF == ord('q'):
    #        break



        # display source image
    #s.DrawBuffer(dispId0, img, last_pic_nr, win_name_img)
    #cv2.destroyAllWindows()
    #s.CloseDisplay(dispId0)
