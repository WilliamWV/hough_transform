import cv2
import numpy as np
import argparse

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


# Main pipeline functions

def parse_input_img():
    parser = argparse.ArgumentParser(description='Receive image')
    parser.add_argument('-i', '--img', required=True, type=str, help='Input image path')
    args = parser.parse_args()
    img = cv2.imread(args.img)
    return img


def filter_image(img):
    # Convert to greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    erase_width = 20
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
    draw__hough_line(img, first_axis, (0, 255, 0), axis_width)
    draw__hough_line(img, second_axis, (0, 255, 0), axis_width)


if __name__ == '__main__':
    img = parse_input_img()
    filtered = filter_image(img)
    first_axis, second_axis = get_axis(filtered)
    erase_axis_points(filtered, first_axis, second_axis)
    adjust_axis(first_axis, second_axis)
    draw_axis(img, first_axis, second_axis)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
