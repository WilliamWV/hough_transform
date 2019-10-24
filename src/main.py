import cv2
import numpy as np

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


img_path = '../exemplo3.jpg'
rho_precision = 1
theta_precision = 2 * np.pi/180

img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
kernel = np.ones((5,5), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=1)
cv2.imshow("Dilated", dilated)


lines = cv2.HoughLines(dilated, rho_precision, theta_precision, 150)
first_axis = []
second_axis = []

axis_thresh = np.pi/20.0

l = 0
while len(first_axis) == 0 or len(second_axis) == 0:
    for rho, theta in lines[l]:
        print("Rho: " + str(rho) + "; Theta: " + str(theta))
        if l == 0:
            first_axis = [rho, theta]
            print("First axis = " + str(first_axis))
        elif abs(abs(first_axis[1] - theta) - np.pi/2.0) < axis_thresh:
            second_axis = [rho, theta]
            print("Second axis = " + str(second_axis))

    l+=1

x, y = cross_point(first_axis, second_axis)

draw__hough_line(dilated, first_axis, (0, 0, 0), 20)
draw__hough_line(dilated, second_axis, (0, 0, 0), 20)

draw__hough_line(img, first_axis, (0, 0, 255), 2)
draw__hough_line(img, second_axis, (0, 0, 255), 2)

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




print("Smaller: " + str(smaller))
print("Bigger: " + str(bigger))

print("First: " + str(first_axis))
print("Second: " + str(second_axis))


draw__hough_line(img, first_axis, (0, 255, 0), 2)
draw__hough_line(img, second_axis, (0, 255, 0), 2)


cv2.imshow("Erased axis", dilated)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
