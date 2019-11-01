import cv2
import numpy as np
import argparse
from collections import defaultdict
import imutils

# Usage:
# $ python main.py -i img_path
# ex:
# $ python main.py -i ../exemplo1.jpg


# Auxiliar functions

# Draw a line parametrized as: rho = x*cos(theta) + y*cos(theta)
# line = [rho, theta]
def draw__hough_line(image, line, color, width):
    theta = line[1]
    rho = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(image, (x1, y1), (x2, y2), color, width)


# Find the point where two lines cross each other
def cross_point(hough_1, hough_2):
    rho1 = hough_1[0]
    theta1 = hough_1[1]
    rho2 = hough_2[0]
    theta2 = hough_2[1]

    a = rho1 * np.cos(theta2) / np.cos(theta1)
    b = np.sin(theta2) - np.sin(theta1)*np.cos(theta2)/np.cos(theta1)
    y = (rho2 - a) / b
    x = (rho1 - y * np.sin(theta1))/np.cos(theta1)
    return x, y

def int_cross_point(hough_1, hough_2):
    cross_p = cross_point(hough_1, hough_2)
    return int(cross_p[0]), int(cross_p[1])

# Main pipeline functions

def parse_input_img():
    parser = argparse.ArgumentParser(description='Receive image')
    parser.add_argument('-i', '--img', required=True, type=str, help='Input image path')
    args = parser.parse_args()
    img = cv2.imread(args.img)
    return img


def filter_image(img):
    # Low pass filter
    blur = cv2.blur(img,(3,3))
    # Convert to greyscale
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    # Get edge map
    edges = cv2.Canny(gray, 100, 200, apertureSize=3)
    # Dilate edge map to fill holes
    dilatation_kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(edges, dilatation_kernel, iterations=1)
    return dilated


def get_axis(img):
    # precisions of the Hough transform
    rho_precision = 1
    theta_precision = 2 * np.pi/180
    # min number of votes to be considered a line
    votes_thresh = 150
    lines = cv2.HoughLines(img, rho_precision, theta_precision, votes_thresh)
    first_axis = []
    second_axis = []

    # angle tolerance between the axis
    axis_thresh = np.pi/20.0

    l = 0
    while len(first_axis) == 0 or len(second_axis) == 0 or l >= len(lines):
        for rho, theta in lines[l]:
            # consider that the most voted line must be one of the axis
            if l == 0:
                first_axis = [rho, theta]
            elif abs(abs(first_axis[1] - theta) - np.pi/2.0) < axis_thresh:
                second_axis = [rho, theta]
        l+=1

    if l >= len(lines):
        print("Couldn't find the axis")
        exit()

    return first_axis, second_axis


# Draw black lines on the filtered image to erase the influence of the axis
# on further processing
def erase_axis_points(img, first_axis, second_axis):
    erase_width = 30
    draw__hough_line(img, first_axis, (0, 0, 0), erase_width)
    draw__hough_line(img, second_axis, (0, 0, 0), erase_width)

# slightly change the axis so that they are orthogonals and maintain the same
# crossing point
def adjust_axis(first_axis, second_axis):
    x, y = cross_point(first_axis, second_axis)
    angle_diff = abs(first_axis[1] - second_axis[1])
    angle_change = np.pi/2.0 - angle_diff
    smaller = first_axis
    bigger = second_axis
    if first_axis[1] > second_axis[1]:
        smaller = second_axis
        bigger = first_axis

    smaller[1] -= angle_change/2
    bigger[1] += angle_change/2

    first_axis[0] = x * np.cos(first_axis[1]) + y * np.sin(first_axis[1])
    second_axis[0] = x * np.cos(second_axis[1]) + y * np.sin(second_axis[1])


def draw_axis(img, first_axis, second_axis):
    axis_width = 2
    draw__hough_line(img, first_axis, (255, 0, 0), axis_width)
    draw__hough_line(img, second_axis, (255, 0, 0), axis_width)

def draw_parabola(img, first_axis, second_axis, xo, yo, p, rotation):
    rad_angle = np.pi * rotation / 180.0
    xorigin, yorigin = int_cross_point(first_axis, second_axis)
    width = img.shape[1]
    height = img.shape[0]

    points = []
    for x in range(-xorigin, width - xorigin):
        try: # Catch division by 0
            y = int(((x-xo)**2) / (4*p)) - yo
            xpixel, ypixel = x + xorigin, y + yorigin
            points.append((xpixel, ypixel))
        except:
            pass



    points = [
        (
            int((point[0] - width/2)* np.cos(rad_angle) - (point[1] - height/2)*np.sin(rad_angle) + width/2) ,
            int((point[0] - width/2) * np.sin(rad_angle) + (point[1] - height/2)*np.cos(rad_angle) + height/2)
        )
        for point in points
    ]

    for point in points:
        cv2.circle(img, (point[0], point[1]), 3, (0, 255, 0), 3)


def find_most_voted_parabola(acc, height, width):
    highest_votes = 0

    for x in range(width):
        for y in range(height):
            dic = acc[x][y]
            for p, votes in dic.items():
                if votes > highest_votes:
                    highest_votes = votes
                    most_voted = [x, y, p]

    return most_voted

def hough_parabola(img, p_precision, downscale_ratio):
    # Downscale the image
    img = cv2.resize(img, (0,0), fx=downscale_ratio, fy=downscale_ratio)
    p_precision *= downscale_ratio

    height, width = img.shape[:2]
    acc = [[defaultdict(lambda: 0) for x in range(height)] for y in range(width)]

    # For each white pixel on filtered image (x, y)
    for x in range(width):
        for y in range(height):
            if img[y, x] != 0:
                # For each possible parabola starting point (xo, yo)
                for xo in range(width):
                    for yo in range(height):
                        try: # To catch division by 0
                            # Calculate p for the given starting point (xo, yo)
                            p = (x - xo)**2 / (4 * (y - yo))
                            p = round(p / p_precision) * p_precision

                            if abs(p) > 2:
                                acc[xo][yo][p] += 1
                        except:
                            pass

    best_parabola = find_most_voted_parabola(acc, height, width)

    # Upscale parabola to fit on the original image
    best_parabola[0] = int(best_parabola[0] / downscale_ratio)
    best_parabola[1] = int(best_parabola[1] / downscale_ratio)
    best_parabola[2] = best_parabola[2] / downscale_ratio

    return best_parabola

def get_parabola(img, first_axis, second_axis):
    x, y, p = hough_parabola(img, 0.01, 0.2)
    xorigin, yorigin = int_cross_point(first_axis, second_axis)

    x = x - xorigin
    y = yorigin - y

    return x, y, p

def get_angles_to_rotate(first_axis, second_axis):
    horizontal = first_axis[1]
    if first_axis[1] - second_axis[1] == np.pi/2:
        horizontal = second_axis[1]

    return - (np.pi/2 - horizontal) * (180/np.pi)

if __name__ == '__main__':
    img = parse_input_img()
    filtered = filter_image(img)
    # Axis handling
    first_axis, second_axis = get_axis(filtered)
    erase_axis_points(filtered, first_axis, second_axis)
    adjust_axis(first_axis, second_axis)
    draw_axis(img, first_axis, second_axis)
    # Rotation handling
    angles = get_angles_to_rotate(first_axis, second_axis)
    rotated = imutils.rotate(filtered, angles)
    # Parabola handling
    x, y, p = get_parabola(rotated, first_axis, second_axis)
    draw_parabola(img, first_axis, second_axis, x, y, p, angles)
    # Show result image
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
