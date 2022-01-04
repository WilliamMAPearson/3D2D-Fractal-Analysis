import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


ImageLocation = "XCT1De-NoiseSegmented - Only Surface.jpg"
print(ImageLocation)
ColourImg = cv2.imread(ImageLocation)
GreyImg = cv2.imread(ImageLocation, 0)


width = min(GreyImg.shape)
length = max(GreyImg.shape)



boxSizeIncrement = 400
currentIteration = boxSizeIncrement
while currentIteration < width:

    cv2.line(ColourImg, (0, currentIteration), (GreyImg.shape[1], currentIteration), (255, 0, 0), thickness=2)
    #cv2.line(ColourImg, (currentIteration, 0), (currentIteration, GreyImg.shape[0]), (255, 0, 0), thickness=2)

    currentIteration = currentIteration + boxSizeIncrement
    

currentIteration = boxSizeIncrement
while currentIteration < length:

    #cv2.line(ColourImg, (0, currentIteration), (GreyImg.shape[1], currentIteration), (255, 0, 0), thickness=2)
    cv2.line(ColourImg, (currentIteration, 0), (currentIteration, GreyImg.shape[0]), (255, 0, 0), thickness=2)

    currentIteration = currentIteration + boxSizeIncrement

cv2.imshow('Colour Image', ColourImg)
plt.imsave('FractalShowcase/Box10.jpg', ColourImg)
