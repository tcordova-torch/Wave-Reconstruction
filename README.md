# Wave-Reconstruction
All files for wavefront reconstruct project
#############################################################
# alignLens

This script has functionality to load a SiSo board, run applet,
aquire a frame, check for lenslet array alignment, adjust it, 
and finally perform zonal reconstruction algorithm. Runs and plots
without additional input. Look through main for appropriate settings
if need to be changed. Will get cleaner version eventually.

Dependencies:
SDZ_v4.py -- centroid calculation and manipulation for ZR math
zonalReconstruct.py -- matrix calculations for ZR
SiSoPyInterface.py -- python wrapper for SiSo functions
numpy -- array, and functionality usage
cv2 -- image loading and manipulation
math -- isclose function
matplotlib -- if plotting images desired
#############################################################
# SDZ_v4.py

This script takes in an image or image array and finds centroids and
shapes it appropiately for ZR computations. Also can handle any plotting,
saving or exporting of ZR output matrices.

Dependencies:
zonalReconstruct.py -- matrix calculations for ZR
SiSoPyInterface.py -- python wrapper for SiSo functions
numpy -- array, and functionality usage
cv2 -- image loading and manipulation
eigen -- matrix math for ZR
math -- isclose function
matplotlib -- plotting 3D output if desired
#############################################################
# SiSoPyInterface.py

Python wrapper for SiSo's C functionality. 
Requires _SiSoPyRt_36.so

#############################################################
# jaiGrabberv3.py

stitched together code to setup siso board connection, grab frame and
calculate angle of lenslet array and plot/display information and 
rotated image.

Dependencies:
SiSoPyInterface.py -- python wrapper for SiSo functions
numpy -- array, and functionality usage
cv2 -- image loading and manipulation
eigen -- matrix math for ZR
math -- isclose function
matplotlib -- plotting 3D output if desired
scipy stats -- linear regression for angle calc
##############################################################

Additional files and versions will be added as project continues
