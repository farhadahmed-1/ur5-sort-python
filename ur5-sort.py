# Import necessary libraries
import urllib.request

import cv2
import numpy as np
import urx
from skimage.transform import ProjectiveTransform
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper

# Initialize tunable parameters
simulate = 0
a = 2
v = 4
snap_pose = [0.5391011732241948, 0.08254341392115624, 0.7281967990628544, -1.8831145852993731, 1.9806709026496,
             -0.5337675885096753]
target_r = [0.5090937784834637, 0.5017593717053667, 0.4256234524477258, -1.8831145852993731, 1.9806709026496,
            -0.5337675885096753]
target_y = [0.2558319064150231, 0.7189667206277992, 0.4256234524477258, -1.8831145852993731, 1.9806709026496,
            -0.5337675885096753]
target_g = [0.1178790777576236, 0.46656833910918244, 0.4256234524477258, -1.8831145852993731, 1.9806709026496,
            -0.5337675885096753]
grab_pose = [0.5391011732241948, 0.08254341392115624, 0.4256234524477258, -1.8831145852993731, 1.9806709026496,
             -0.5337675885096753]
#i_tl = [43, 85]
#i_tr = [37, 548]
#i_bl = [443, 92]
#i_br = [438, 547]
#i_tl = [67, 92]
#i_tr = [63, 546]
#i_bl = [434, 95]
#i_br = [440, 558]
i_tl = [81.5, 83.5]
i_tr = [79, 569.5]
i_bl = [430.5, 93.5]
i_br = [422.0, 566.0]
i_c = [252, 316]
src = np.asarray([i_bl, i_tl, i_c, i_tr, i_br])
#w_tl = [.285, -0.098]
#w_tr = [.297, .287]
#w_bl = [.616, -0.106]
#w_br = [.627, .278]
#w_tl = [0.29416068099848053, -0.11106045244674607]
#w_tr = [0.30323492653835876, 0.2674049203549436]
#w_bl = [0.6050979989774012, -0.1233051699920784]
#w_br = [0.6259044618009142, 0.2658443178067092]
w_tl = [0.3090843777170824, -0.10271466299765152]
w_tr = [0.3239865023210112, 0.30777591465563725]
w_bl = [0.6082008596507158, -0.1098611529665721]
w_br = [0.6203702963952806, 0.2953936090848621]
w_c = [0.4622043240148198, 0.07025897657195562]
dst = np.asarray([w_bl, w_tl, w_c, w_tr, w_br])
l_s = 170
u_s = 255
l_v = 50
u_v = 255
l_b_r1 = np.array([0, l_s, l_v])
u_b_r1 = np.array([4, u_s, u_v])
l_b_r2 = np.array([174, l_s, l_v])
u_b_r2 = np.array([180, u_s, u_v])
l_b_y = np.array([21, l_s, l_v])
u_b_y = np.array([42, u_s, u_v])
l_b_g = np.array([54, l_s, l_v])
u_b_g = np.array([86, u_s, u_v])
b_0 = np.array([0, 0, 0])


# Function for finding image and world coordinates of the objects by iterating through contours
def coord(contours, r, g, b, box_img):
    t = ProjectiveTransform()
    t.estimate(src, dst)
    i_nos = 0
    objects = []
    for k in contours:
        y_k, x_k, w_k, h_k = cv2.boundingRect(k)
        if w_k > 40 and h_k > 40:
            cv2.rectangle(box_img, (y_k, x_k), (y_k + w_k, x_k + h_k), (b, g, r), 2)
            objects.append([x_k + h_k / 2, y_k + w_k / 2])
            i_nos += 1
    if(i_nos==0):
        world = []
    else:
        world = t(objects)
    return i_nos, objects, world


# Function for finding contours using HSV Bounds
def cont(l_b_1, u_b_1, l_b_2, u_b_2, hsv_img, image):
    mask = cv2.bitwise_or(cv2.inRange(hsv_img, l_b_1, u_b_1), cv2.inRange(hsv_img, l_b_2, u_b_2))
    res = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(blurred, 0, 255, 1)
    dilate = cv2.dilate(canny, np.ones((5, 5), np.uint8), iterations=1)
    contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    return contours


def snapshot():
    rob.movel(snap_pose, a, v)
    print("Snap Pose ", snap_pose, " reached")
    pull_image = urllib.request.urlopen("http://192.168.1.6:4242/current.jpg?type=color")
    image = cv2.imdecode(np.asarray(bytearray(pull_image.read()), dtype="uint8"), cv2.IMREAD_COLOR)
    print("Snapshot taken")
    # Convert to HSV, create a copy of the Image and prepare Transformation Matrix
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    box_img = image
    # Find Contours of R, Y and G objects in the image
    contours_r = cont(l_b_r1, u_b_r1, l_b_r2, u_b_r2, hsv_img, image)
    contours_y = cont(l_b_y, u_b_y, b_0, b_0, hsv_img, image)
    contours_g = cont(l_b_g, u_b_g, b_0, b_0, hsv_img, image)
    # Iterate thorough contours to find the coordinates of R, Y and G objects
    i_r, r_objects, r_world = coord(contours_r, 255, 0, 0, box_img)
    i_y, y_objects, y_world = coord(contours_y, 200, 200, 0, box_img)
    i_g, g_objects, g_world = coord(contours_g, 0, 255, 0, box_img)
    return box_img, r_world, y_world, g_world

def print_image(box_img):
    # Print Image with boxes on the screen
    cv2.namedWindow(winname="box_img", flags=cv2.WINDOW_NORMAL)
    cv2.imshow(winname="box_img", mat=box_img)
    cv2.waitKey(delay=1)

    # Function for Pick and Place operation
def pick_place(world, target, color):
    i = 0
    for _ in world:
        print(color, " object no. ", i + 1)
        grab_pose[0] = world[i][0]
        grab_pose[1] = world[i][1]
        grab_pose[2] = 0.4256234524477258
        if simulate == 0:
            rob.movel(grab_pose, a, v)
        #print("Grab Pose H1 ", grab_pose, " reached")
        grab_pose[2] -= .15
        if simulate == 0:
            rob.movel(grab_pose, a, v)
        #print("Grab Pose H0 ", grab_pose, " reached")
        if simulate == 0:
            r_grip.close_gripper()
        #print("Gripper Closed")
        grab_pose[2] += .15
        if simulate == 0:
            rob.movel(grab_pose, a, v)
        #print("Grab Pose H1 ", grab_pose, " reached")
        if simulate == 0:
            rob.movel(target, a, v)
        #print("Target ", target, " reached")
        if simulate == 0:
            r_grip.open_gripper()
        print("Gripper Opened")
        i += 1

# Initialize Robot and take a snapshot
rob = urx.Robot("192.168.1.6")
print("Robot initialized at 192.168.1.6")
r_grip = Robotiq_Two_Finger_Gripper(rob)
print("Gripper initialized")
r_grip.open_gripper()
rob.movej((-1.96, -1.53, 1.58, -2.12, -1.56, 1.19), a, v)
print("Start position reached")

while True:
    s = input("Enter Command: ")
    if(s == 'h'):
        print("Help:\n")
        print("'r' - Run pick and place program\n")
        print("'t' - Teach robot new colour")

    elif(s == 'r'):
        image, r_world, y_world, g_world = snapshot()
        print_image(image)
        # Carry out Pick and Place for R, Y and G objects
        pick_place(r_world, target_r, "Red")
        pick_place(y_world, target_y, "Yellow")
        pick_place(g_world, target_g, "Green")
        rob.movel(snap_pose, a, v)
        print("Pick and place complete")


    elif(s == 't'):
        print("Teach")

# Bring robot back to snap position and then exit control
if simulate == 0:
    rob = urx.Robot("192.168.1.6")
    rob.movel(snap_pose, a, v)
    #print("Snap Pose ", snap_pose, " reached")
    rob.close()
    print("Control Ended")
else:
    #print("Snap Pose ", snap_pose, " reached")
    print("Control Ended")


# Wait for Image Window to be closed and then exit program
print("Press Escape on Image Window")
while True:
    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyAllWindows()
exit()
