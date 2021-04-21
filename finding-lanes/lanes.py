import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

# gaussian blur, hough space/transform, bit masking 

def make_coordinates(image, line_parameters): 
    slope, intercept = line_parameters
    y1 = image.shape[0] 
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines): 
    left_fit = [] # left side of lane
    right_fit = [] # right side of lane 
    for line in lines: 
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        slope = parameters[0] 
        intercept = parameters[1] 
        if slope < 0: # y axis is flipped for images, and so is slope 
            left_fit.append((slope, intercept)) 
        else: 
            right_fit.append((slope, intercept))
    if len(left_fit) and len(right_fit):
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)
        right_line = make_coordinates(image, right_fit_average)
        return np.array([left_line, right_line])

def canny(image): 
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # convert to grayscale 
    blur = cv2.GaussianBlur(gray, (5,5), 0) # gaussian blur 
    canny = cv2.Canny(blur, 50, 150) # computes derivate of change in x and y direction and if ratio passes threshold, shows in image
    return canny 

def display_lines(image, lines): 
    line_image = np.zeros_like(image) # empty image 
    if lines is not None: 
        for x1, y1, x2, y2 in lines: 
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10) # draw lines on empty image 
    return line_image


def region_of_interest(image): 
    height = image.shape[0] # grab height of image
    polygons = np.array([[(200,height), (1100,height), (550,250)]]) # create triangular shape 
    mask = np.zeros_like(image) # fill a new image of same size with all 0s
    cv2.fillPoly(mask,polygons, 255) # create mask 
    masked_image = cv2.bitwise_and(image,mask) # mask the image with the the mask just created, bitwise and makes all regions not in mask 0
    return masked_image

# image = cv2.imread('test_image.jpg') # read in test image 
# lane_image = np.copy(image) # copy of image 
# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) # hough transform given coordinates and thresholds
# averaged_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, averaged_lines)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1) # weighted average sum of images 
# cv2.imshow('result',combo_image)
# cv2.waitKey(0)

cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()): 
    ret, frame = cap.read()
    if ret: 
        canny_image = canny(frame)
        cropped_image = region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) # hough transform given coordinates and thresholds
        averaged_lines = average_slope_intercept(frame, lines)
        line_image = display_lines(frame, averaged_lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1) # weighted average sum of images 
        cv2.imshow('result',combo_image)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    else: 
        break
cap.release() 
cv2.destroyAllWindows()

