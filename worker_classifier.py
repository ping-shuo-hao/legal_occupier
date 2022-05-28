import cv2
import numpy as np
import random as rng


text_color = (0, 0, 255)



#green=0.2,red=0.25
# define the range of color of interest in HSV, according to https://stackoverflow.com/questions/51229126/how-to-find-the-red-color-regions-using-opencv
lowergreen = np.array([30, 0,200])
uppergreen = np.array([90,180,255])
lowerred1 = np.array([0, 50, 20])
upperred1 = np.array([5, 255, 255])
lowerred2 = np.array([175,50,20])
upperred2 = np.array([180,255,255])

def classifer(img, thres_green=0.15,thres_red=0.15, draw=False):
    # img -- worker image
    # threshold -- threshold define experimentally for COI percentage
    # draw -- var contorl whether darw intermediate results 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Threshold the HSV image to get only color of interest
    maskgreen = cv2.inRange(hsv, lowergreen, uppergreen)
    maskred1=cv2.inRange(hsv,lowerred1,upperred1)
    maskred2=cv2.inRange(hsv,lowerred2,upperred2)
    # Bitwise-AND mask and original image
    resgreen = cv2.bitwise_and(img, img, mask=maskgreen)
    resred1=cv2.bitwise_and(img,img,mask=maskred1)
    resred2=cv2.bitwise_and(img,img,mask=maskred2)
    resred=cv2.bitwise_or(resred1,resred2)
#    cv2.imwrite('res.jpg',resgreen)
    # calculate the the percentage of COI pixels
    ROI_colored_rate_green = np.sum(np.where(resgreen != 0, 1, 0)) / (img.shape[0] * img.shape[1]) / 3
    ROI_colored_rate_red = np.sum(np.where(resred != 0, 1, 0)) / (img.shape[0] * img.shape[1]) / 3
    if draw:
        print("ROI_colored_rate_green", ROI_colored_rate_green)
        print("ROI_colored_rate_red", ROI_colored_rate_red)
#        cv2.imwrite('res.jpg', res)
#        cv2.imwrite('img.jpg', img)
#    cv2.waitKey(0)
    return (ROI_colored_rate_green > thres_green or ROI_colored_rate_red>thres_red)

if __name__ == "__main__":
    im = cv2.imread('worker.png', 3)
    classifer(im)