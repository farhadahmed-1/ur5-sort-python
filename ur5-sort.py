# Import necessary libraries

import urx                          # 3rd party library for controlling the UR5
import cv2                          # Open CV for image processing
import urllib.request               # Library for grabbing images from Wrist Camera URL
import numpy as np                  # NumPy for array operations (to compliment OpenCV)
from skimage.transform import ProjectiveTransform
                                    # Projective Transform for Image to World Coordinate Transformation
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
                                    # Two Finger Gripper library


# Initialize Global / Main Variables (tunable parameters)

simulate = 0                        # Robot only moves if simulate == 0
a = 2                               # Acceleration
v = 4                               # Velocity
snap_pose = [0.5391011732241948, 0.08254341392115624, 0.7281967990628544, -1.8831145852993731, 1.9806709026496,
             -0.5337675885096753]   # Fixed Pose for taking Snapshots
grab_pose = [0.5391011732241948, 0.08254341392115624, 0.4256234524477258, -1.8831145852993731, 1.9806709026496,
             -0.5337675885096753]   # Similar Pose as snap_pose but with a lower height, altered later to grab objects
teach_pose = [0.09874399087472048, 0.53439499719666, 0.4674974138741603, -1.8831145852993731, 1.9806709026496,
              -0.5337675885096753]  # Fixed Pose for teaching a new colour
col_name = ["Red", "Yellow", "Green"]
                                    # Array for storing list of colors - By default, [0] is Red, [1] is Yellow and
                                    # [2] is Green. Additional colours can be added from [3] onwards
target = [[0.5090937784834637, 0.5017593717053667, 0.4256234524477258, -1.8831145852993731, 1.9806709026496,
           -0.5337675885096753],
          [0.2558319064150231, 0.7189667206277992, 0.4256234524477258, -1.8831145852993731, 1.9806709026496,
           -0.5337675885096753],
          [0.1178790777576236, 0.46656833910918244, 0.4256234524477258, -1.8831145852993731, 1.9806709026496,
           -0.5337675885096753]]    # Array to store the default bucket positions in the same order as Color Name array

# Image Reference Points (Top-Left, Top-Right, Bottom-Left, Bottom-Right and Centre)
i_tl = [81.5, 83.5]
i_tr = [79, 569.5]
i_bl = [430.5, 93.5]
i_br = [422.0, 566.0]
i_c = [247.5, 331.5]

# Use these values as Image Reference Points for Low Res Pictures
# i_tl = [81.5, 83.5]
# i_tr = [79, 569.5]
# i_bl = [430.5, 93.5]
# i_br = [422.0, 566.0]
# i_c = [247.5, 331.5]

# World Reference Points (Top-Left, Top-Right, Bottom-Left, Bottom-Right and Centre)
w_tl = [0.3090843777170824, -0.10271466299765152]
w_tr = [0.3239865023210112, 0.30277591465563725]
w_bl = [0.5982008596507158, -0.1068611529665721]
w_br = [0.6103702963952806, 0.2903936090848621]
w_c = [0.45448932497975086, 0.10681846161071724]

# Use these values as World Reference Points for Low Res Pictures
# w_tl = [0.3090843777170824, -0.07771466299765152]
# w_tr = [0.3239865023210112, 0.43277591465563725]
# w_bl = [0.6782008596507158, -0.0818611529665721]
# w_br = [0.6803702963952806, 0.4203936090848621]
# w_c = [0.45448932497975086, 0.10681846161071724]

# Lower and Upper Saturation and Value Thresholds for Image Processing
l_s = 170
u_s = 255
l_v = 50
u_v = 255

#  Arrays to store Upper and Lower Bounds for Image Processing in the order of Colors as in the Colour Name array
l_b_1 = [[0, l_s, l_v], [21, l_s, l_v], [54, l_s, l_v]]
u_b_1 = [[4, u_s, u_v], [42, u_s, u_v], [86, u_s, u_v]]

# Additional Bounds to take care of "Red" [0] colour overflowing across the 180 - 0 border
l_b_2 = [[174, l_s, l_v], [0, 0, 0], [0, 0, 0]]
u_b_2 = [[180, u_s, u_v], [0, 0, 0], [0, 0, 0]]


# Define all functions

# Function for finding contours using HSV Bounds
def cont(cont_l_b_1, cont_u_b_1, cont_l_b_2, cont_u_b_2, hsv_img, cont_image):

    # Convert the Upper and Lower Bounds into NumPy Arrays for processing through OpenCV
    lb1 = np.array(cont_l_b_1)
    ub1 = np.array(cont_u_b_1)
    lb2 = np.array(cont_l_b_2)
    ub2 = np.array(cont_u_b_2)

    mask = cv2.bitwise_or(cv2.inRange(hsv_img, lb1, ub1), cv2.inRange(hsv_img, lb2, ub2))
                                    # Create a Mask with only the pixels in the selected HSV Range
    res = cv2.bitwise_and(cont_image, cont_image, mask=mask)
                                    # Apply the mask on the original image - not necessary for our processing
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
                                    # Convert the masked image into Grayscale
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                                    # Blur the image to fill gaps and avoid separate object detection for single object
    canny = cv2.Canny(blurred, 0, 255, 1)
                                    # Canny Edge Detection
    dilate = cv2.dilate(canny, np.ones((5, 5), np.uint8), iterations=1)
                                    # Dilate the image to again fill gaps, if any, after Canny Edge Detection
    contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    # Identify separate contours in the image and save to an array
    contours = contours[0] if len(contours) == 2 else contours[1]
                                    # Depending on OpenCV version, pick the correct argument returned by cv.findContours
    return contours                 # Return the array of contours of the specified color


# Function for finding image and world coordinates of the objects by iterating through contours
def coord(contours, r, g, b, box_img):
    t = ProjectiveTransform()       # Initiate Projective Transform

    # Use the Image and World Reference Points to Generate Source and Destination NumPy arrays for the transform
    src = np.asarray([i_bl, i_tl, i_tr, i_br])
    dst = np.asarray([w_bl, w_tl, w_tr, w_br])

    t.estimate(src, dst)            # Prepare the transform model
    i_nos = 0                       # Count number of objects of the color
    objects = []                    # Start with an empty array for image coordinates
    for k_coord in contours:        # Check each contour for a valid object
        y_k, x_k, w_k, h_k = cv2.boundingRect(k_coord)
                                    # Bound the individual contour with a rectangle
        if w_k > 40 and h_k > 40:   # Only select objects bigger than a threshold - to filter out disturbances
            cv2.rectangle(box_img, (y_k, x_k), (y_k + w_k, x_k + h_k), (b, g, r), 2)
                                    # Draw a box on the image - to be subsequently printed on screen
            objects.append([x_k + h_k / 2, y_k + w_k / 2])
                                    # Add the object to the array (image coordinates)
            i_nos += 1              # Increase the count of number of objects
    if i_nos == 0:                  # If there are no valid objects in the image
        world = []                  # return empty array for world coordinates
    else:                           # If there are valid objects in the image
        world = t(objects)          # Carry out Projective Transform to populate the world coordinates array
    return i_nos, objects, world    # return the no. of objects, image coordinates and world coordinates


# Function to Print Image with boxes on the screen
def print_image(box_img):
    cv2.namedWindow(winname="box_img", flags=cv2.WINDOW_NORMAL)
                                    # Create a CV2 Image Display window
    cv2.imshow(winname="box_img", mat=box_img)
                                    # Print the image with objects bounded in rectangles
    cv2.waitKey(delay=1)            # Non-blocking command


# Function for Pick and Place operation of a certain colour objects
def pick_place(world, pp_target, color):
    print("Pick and place commencing on " + color + " objects.")
    i = 0                           # Count for index no
    for _ in world:                 # For as many objects as in the world array
        print(color, " object no. ", i + 1)

        # Set the grab pose vertically above the object
        grab_pose[0] = world[i][0]
        grab_pose[1] = world[i][1]
        grab_pose[2] = 0.4256234524477258

        if simulate == 0:
            rob.movel(grab_pose, a, v)
                                    # Move to Grab Pose
        grab_pose[2] -= .15         # Reduce Height to grab object
        if simulate == 0:
            rob.movel(grab_pose, a, v)
                                    # Move to the reduced height
            r_grip.close_gripper()  # Close the Gripper
        grab_pose[2] += .15         # Increase Height so that other objects are not disturbed
        if simulate == 0:
            rob.movel(grab_pose, a, v)
                                    # Move to the original Grab Pose height
            rob.movel(pp_target, a, v)
                                    # Move to the Target Bin Location
            r_grip.open_gripper()   # Open the Gripper to drop the object into the bin
        i += 1                      # Increase the complete count
    print("Pick and place complete on " + color + " objects.")


# Function for the Full Pick and Place Operation from Start to End
def full_pp():
    if simulate == 0:
        rob.movel(snap_pose, a, v)  # Move to the Snap Pose (if not already moved)
    print("Snap Pose reached")
    if simulate == 0:
        pull_image = urllib.request.urlopen("http://192.168.1.6:4242/current.jpg?type=color")
                                    # Pull Image from the Robotiq Wrist Camera URL
        snap_image = cv2.imdecode(np.asarray(bytearray(pull_image.read()), dtype="uint8"), cv2.IMREAD_COLOR)
                                    # Convert image to a NumPy array for CV2 processing
    else:
        snap_image = cv2.imread('current (1).jpg')
    print("Snapshot taken")
    hsv_img = cv2.cvtColor(snap_image, cv2.COLOR_BGR2HSV)
                                    # Convert to HSV
    box_img = snap_image            # Create a copy of the Image for printing with boxes
    contours = []                   # Start with an empty Contours Master array
    k_cont = 0                      # Keep Count of the Colors processed
    for _ in col_name:              # Process the image using every Color in the Color Name array
        xd = cont(l_b_1[k_cont], u_b_1[k_cont], l_b_2[k_cont], u_b_2[k_cont], hsv_img, snap_image)
                                    # Get the contours for the particular color
        contours.insert(k_cont, xd)
                                    # Append the contours to Contours Master array
        k_cont += 1                 # Increase Count of Colors processed
    i_nos = []                      # Initialize empty array to store color wise no of objects
    objects_img = []                # Initialize empty array to store color wise object Image Coordinates
    objects_world = []              # Initialize empty array to store color wise object World Coordinates
    k_cont = 0                      # Keep Count of the Colours processed
    for _ in col_name:              # Iterate through Contours of Each Color in the Color Name array
        col_hsv = np.uint8([[[(l_b_1[k_cont][0] + u_b_1[k_cont][0]) / 2, 213, 153]]])
                                    # Get the average hue of the color range for the bounding box
        col_rgb = cv2.cvtColor(col_hsv, cv2.COLOR_HSV2BGR)
                                    # Convert the average hue to RGB to send to function
        xa, xb, xc = coord(contours[k_cont], int(col_rgb[0][0][2]), int(col_rgb[0][0][1]), int(col_rgb[0][0][0]),
                           box_img)
                                    # Store the no of objects, Image Coordinates and World Coordinates of the objects
        i_nos.insert(k_cont, xa)    # Save the no. of objects for the particular colour
        objects_img.insert(k_cont, xb)
                                    # Save the image coordinates for the particular colour
        objects_world.insert(k_cont, xc)
                                    # Save the world coordinates for the particular colour
        k_cont += 1                 # Increase the Count of the Colours processed
    print_image(box_img)            # Print the image on the screen once all objects are detected
    k_cont = 0                      # Keep count of the Colours processed
    for _ in col_name:              # Iterate through all the available colours
        pick_place(objects_world[k_cont], target[k_cont], col_name[k_cont])
                                    # Send object and bin co-ordinates and let sub-function carry out Pick and Place
        k_cont += 1                 # Increase the Count of the Colours processed
    if simulate == 0:
        rob.movel(snap_pose, a, v)  # Move robot arm back to Snap Pose
    print("Pick and place complete. Press Escape on Image Window to return control.")
    while True:                     # Wait for Image Window to be closed
        key = cv2.waitKey(1)        # Check key press in the loop
        if key == 27:               # Key 27 is Escape Key
            break                   # If pressed, break from loop
    cv2.destroyAllWindows()         # Close the Image Window


# Function for manually teaching a new colour
def teach():
    if simulate == 0:
        rob.movel(teach_pose, a, v)
                                    # Move robot arm to the side for teaching a new colour
    input("Place Object below camera and press any key")
    if simulate == 0:
        pull_image = urllib.request.urlopen("http://192.168.1.6:4242/current.jpg?type=color")
                                    # Pull Image from the Robotiq Wrist Camera URL
        snap_image = cv2.imdecode(np.asarray(bytearray(pull_image.read()), dtype="uint8"), cv2.IMREAD_COLOR)
                                    # Convert image to a NumPy array for CV2 processing
    else:
        snap_image = cv2.imread('current (2).jpg')
    print("Snapshot taken")
    hsv_img = cv2.cvtColor(snap_image, cv2.COLOR_BGR2HSV)
                                    # Convert RGB image to HSV
    average = (hsv_img[240][320][0] / 9 + hsv_img[239][320][0] / 9 + hsv_img[241][320][0] / 9 + hsv_img[240][319][
        0] / 9 + hsv_img[239][319][0] / 9 + hsv_img[241][319][0] / 9 + hsv_img[240][321][0] / 9 + hsv_img[239][321][
                   0] / 9 + hsv_img[241][321][0] / 9)
                                    # Get average hue of centre pixels

    # Append lower and upper bounds for the new colour
    l_b_1.append([average - 15, l_s, l_v])
    u_b_1.append([average + 15, u_s, u_v])
    l_b_2.append([0, 0, 0])
    u_b_2.append([0, 0, 0])

    input("Move Arm to Bin Location and press any key")
    if simulate == 0:
        t = rob.getl()              # Get the target bin location
    else:
        t = snap_pose
    target.append([t[0], t[1], 0.4256234524477258, -1.8831145852993731, 1.9806709026496,
                   -0.5337675885096753])
                                    # Append target bin location (x and y) to the array
    s_teach = input("Enter Color Name")
    col_name.append(s_teach)        # Save the name of the new colour
    print("Color", s_teach, " Added")
    if simulate == 0:
        rob.movel(snap_pose, a, v)  # Move robot arm back to Snap Pose


# Function to Redefine all the bin locations manually
def define_bin():
    if simulate == 0:
        rob.movel(teach_pose, a, v)
                                    # Move robot arm to the side for teaching new bin locations
    i = 0                           # Initiate count of colours processed
    for k in col_name:              # For Each Colour
        print("Move Arm to ", k, " Bin Location and press any key")
        input()
        if simulate == 0:
            t = rob.getl()          # Get current arm location
        else:
            t = target[i]
        target[i] = [t[0], t[1], 0.4256234524477258, -1.8831145852993731, 1.9806709026496,
                     -0.5337675885096753]
                                    # Save x and y from current position
        i += 1                      # Increase the Count of the Colours processed
    input("All Bins defined. Please move away from the robot and press any key to continue.")
    if simulate == 0:
        rob.movel(snap_pose, a, v)  # Move robot arm back to Snap Pose


# Main Program Execution

connected = False
while not connected:
    s = input("Enter Command (h for help): ")
    if s == 'c':  # connect
        if simulate == 0:
            while True:
                print("Connecting...")
                try:
                    rob = urx.Robot("192.168.1.6")
                                    # Initialize Robot Control
                    r_grip = Robotiq_Two_Finger_Gripper(rob)
                                    # Initialize Gripper Control
                    r_grip.open_gripper()
                                    # Open Gripper if previously closed
                    rob.movej((-1.96, -1.53, 1.58, -2.12, -1.56, 1.19), a, v)
                                    # Move Robot arm to a safe position
                    rob.movel(snap_pose, a, v)
                                    # Move Robot arm to the Snap Pose
                    print("Robot initialized at 192.168.1.6, Gripper initialized and Snap position reached")
                    connected = True
                    break
                except:
                    s = input("Try again (y/n)?")
                    if s == 'n':
                        break
        else:
            print("Robot initialized at 192.168.1.6, Gripper initialized and Snap position reached")
            break
        if connected:
            break

    if s == 'm':
        s = input("Set mode ('1' for sim, 0 for real): ")
        if s == '1':
            simulate = 1
        if s == '0':
            simulate = 0

    if s == 'h':
        print("Help:")
        print("'c' - connect to UR5")
        print("'m' - set simulation mode")
        print("'q' - quit")

    if s == 'q':
        print("Control Ended")
        quit()

while True:                         # Interactive Menu
    s = input("Enter Command (h for help - command list): ")
    if s == 'h':                    # Help Option
        print("Help:")
        print("'r' - Run pick and place program")
        print("'t' - Teach robot new colour")
        print("'d' - Define new bin locations ")
        print("'q' - Quit")
    elif s == 'r':                  # Run Pick and Place Option
        print("Starting pick and place program")
        full_pp()                   # Call Full Pick and Place Function
    elif s == 't':                  # Teach New Colour Option
        print("Starting Teach function")
        teach()                     # Call Teach Function
    elif s == 'd':                  # Define New Bin Locations Option
        print("Define new bin locations")
        define_bin()                # Call Define Bin Function
    elif s == 'q':                  # Quit Option
        break                       # Break the endless loop

if simulate == 0:
    rob = urx.Robot("192.168.1.6")  # Re-connect robot
    rob.movel(snap_pose, a, v)      # Bring robot back to snap position
    rob.close()                     # Disconnect robot
print("Control Ended.")
# exit()
