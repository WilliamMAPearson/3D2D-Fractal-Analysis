import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


def DrawAGraph(BoxSize, Count):
    plt.scatter(BoxSize, Count)
    plt.xlabel('Log(Box Size)')
    plt.ylabel('Log(Count)')
    plt.show()


def DrawLines(Image, sizes):
    for size in sizes:
        cv2.line(Image, (0, size), (Image.shape[1], size), (0, 255, 255), thickness=1)
        cv2.line(Image, (size, 0), (size, Image.shape[0]), (0, 255, 255), thickness=1)

    cv2.imshow('Colour Image', Image)
    cv2.waitKey(0)


def BoxCount(BinaryImage, boxsize):
    # Create the locations at where the splits will occur, from 0 and every increment upto the height/width of the image
    XArray = np.arange(0, BinaryImage.shape[0], boxsize)
    YArray = np.arange(0, BinaryImage.shape[1], boxsize)

    # Reduce function sums up all the parts of an array in question.
    # These two functions here essentially splits the image into the defined boxes and counts how many pixels are 1 in each box
    XReduceAt = np.add.reduceat(BinaryImage, XArray, axis=0)
    XYReduceAt = np.add.reduceat(XReduceAt, YArray, axis=1)

    # Return two arrays that contain the index of all the boxes that are not zero and also no full.
    # Full is defined as having as many black pixels as there are pixels in a box
    # If a box is zero then it means it contains no black pixels, if a box == the boxsize*boxsize, then it means
    #  that every pixel in that box is 1, so ignores it.
    # Both these functions return the same value.
    Where = np.where((XYReduceAt > 0) & (XYReduceAt < boxsize*boxsize))
    Where = np.asarray((XYReduceAt > 0) & (XYReduceAt < boxsize*boxsize)).nonzero()

    # The Arrays returned are in the format [a, b, c, d, e, f...], [m, n, l, o, p, q...] where a box cordinate is given by [a, m], or [c, l]
    # Therefore the amount of boxes that are both no-zero and also not full equals the amount of boxes to count.
    # So we take the first array (can also take the second it doesnt matter which) and take the length of it.
    count = len(Where[0])
    return count


def CalculateBoxSpaces(BinaryImage, Automatic, AddCustomRange, ManualMin, ManualMax, CustomSizes, CustomSizesList):

    # Returns the minimal dimension of image minimum value of shape(x,y)
    MinimumDimension = min(BinaryImage.shape)
    print('Image Shape', BinaryImage.shape)
    print('Minimum Dimension', MinimumDimension)
    
    # Choose automatic if you want the program to choose spacing based on max and min image size
    if Automatic:
        # Greatest power of 2 less than or equal to MinimumDimension of the image
        n = int(np.floor(np.log(MinimumDimension)/np.log(2)))
        print('Calculated n', n)
        n = n + 1 #append one more
        print('Assigned n', n)

        # Build successive box sizes (from 2**n down to 2**1)
        sizes = 2**np.arange(n, 1, -1)
        sizes = np.append(sizes, 2) #append 2

        # Decide wether to add a custom range of numbers
        if AddCustomRange:
            centrePoints = np.arange(ManualMin, ManualMax)
            sizes = np.append(sizes, centrePoints)
            
    # If not choosing automatic then choose own range.        
    if AddCustomRange:
        sizes = []
        centrePoints = np.arange(ManualMin, ManualMax)
        for i in centrePoints:
            sizes.append(i)

    if CustomSizes:
        sizes = CustomSizesList
        
        
    print('Box Sizes', sizes)
    return sizes
    

def FractalDimension(GreyImage, threshold, ColourImage, Automatic, AddCustomRange, ManualMin, ManualMax, CustomSizes, CustomSizesList):
    # Checks if the image being imported is a 2D image, if it is it will have shape (x, y)
    # Assertion check returns an assertionError if false, else it returns true and allows the code to run.
    assert(len(GreyImage.shape) == 2)

    # Transform the GreyImage into a binary True/False Image
    # If the value is below a threshold value then the pixel value becomes true, else become false
    BinaryImage = (GreyImage < threshold)

    # Calculate Box Spacings
    sizes = CalculateBoxSpaces(BinaryImage, Automatic, AddCustomRange, ManualMin, ManualMax, CustomSizes, CustomSizesList)
    
    #Draw Lines On Image
    DrawLines(ColourImage, sizes)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        count = BoxCount(BinaryImage, size)
        counts.append(count)
        print('size', size, 'Count', count)
        
    # Plot 
    DrawAGraph(np.log(sizes),np.log(counts))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]


ImageLocation = "Julia_z2+0,25.png"
print(ImageLocation)
ColourImg = cv2.imread(ImageLocation)
GrayImg = cv2.imread(ImageLocation, 0)
cv2.imshow('Original Image', GrayImg)
cv2.waitKey(0)
FD = FractalDimension(GrayImg, threshold=150, ColourImage=ColourImg,
                      Automatic=False,
                      AddCustomRange=False, ManualMin=8, ManualMax=512,
                      CustomSizes=True, CustomSizesList=[400, 200, 100, 50, 25, 10])
print("Minkowski–Bouligand dimension (computed): ", FD)
