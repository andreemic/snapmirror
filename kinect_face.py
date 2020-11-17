import cv2
import dlib
import sys
import freenect
import math
import numpy as np
import operator
from time import time


    
''' Definitions:
        Metric space - 3d world space centered around center of screen. Looking from the camera: +y = up, +x = left, +z = forward
             Units: mm
        Raw Metric space - Metric space centered around webcam.
        Frame space - pixel space of 2d picture as seen by webcam centered around top-left corner. Units: px
        Normalized frame space - Frame space centered around picture center.
        Screen space - pixel space of the whole screen centered around top-left corner. Units: px

'''

'''
    CONSTANTS & PARAMETERS
'''
debug = False
image_w = 640 # Size of kinect image
image_h = 480
debug_w = 320 # Size of webcam display in bottom right corner   
debug_h = 240
screen_res = (1680, 1050)
#screen_mm = (490, 295) # Screen size in mm
screen_mm = (510, 380) # Screen size in mm
#cam_screen_offset_y = 30 # Vertical gap between top screen edge and webcam
cam_screen_offset_y = 60 # Vertical gap between top screen edge and webcam

webcam_y = cam_screen_offset_y + int(screen_mm[1] / 2)
webcam_x = 19 # Horizontal gap between kinect middle and webcam lens

# Offsets that are calibrated later
px_offset_x = 0
px_offset_y = 0

# Kinect sensor FOV (see https://openkinect.org/wiki/Imaging_Information)
fov_x = 62
fov_y = 48.6

op_distance = (800, 2000) # Operational distance from/to in mm

'''
    FUNCTIONS
'''
# Receives p in metric space
# Returns point at which p is going to be reflected on the screen in screen space as seen from eye 
def screen_refection(p, viewpoint):
    (px, py, _) = p
    (vx, vy, _) = viewpoint
    
    # Convert reflection coordinates to screen space
    x_screen = (vx + px) / 2 + (screen_mm[0] / 2) # 0 is now left screen end
    y_screen = (vy + py) / 2 + (screen_mm[1] / 2)  # 0 is now top screen end
    
    # To-Do: Detect off screen point and return False

    x_px = math.floor(screen_res[0] * x_screen / screen_mm[0])
    y_px = math.floor(screen_res[1] * y_screen / screen_mm[1])
    
    return (x_px + px_offset_x, y_px + px_offset_y)

# Converts frame coordinates to metric space centered around camera
# see https://openkinect.org/wiki/Imaging_Information
def frame_to_raw_metric(x_px, y_px, z_mm):
    x_mm = (x_px - image_w / 2) * (z_mm - 10) * 0.0021
    y_mm = (y_px - image_h / 2) * (z_mm - 10) * 0.0021

    return (x_mm, y_mm)

# Converts xy-point in frame space to xyz-point in metric space (centered around screen)
# 10px correspond to 1° (https://smeenk.com/kinect-field-of-view-comparison/)
def frame_to_metric(x_px, y_px, z_mm):

    (x_mm, y_mm) = frame_to_raw_metric(x_px, y_px, z_mm)
    
    '''
    (x_norm, y_norm) = normalize_frame_point((x_px, y_px))
    x_deg = x_norm * image_w / 10
    y_deg = y_norm * image_h / 10 #AMBIG.: the 10/1 ratio is not quite accurate for y
    
    x_mm = math.sin(math.radians(x_deg)) * z_mm * -1 # Account for mirror flip
    y_mm = math.sin(math.radians(y_deg)) * z_mm
    '''
    return (-x_mm - webcam_x, y_mm - webcam_y, z_mm);


def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

# point
def reflect_landmarks(landmarks_px, points_z, viewpoint_px, viewpoint_z):
    '''
    Reflect pointsfrom screen as seen from viewpoint.

    Keyword arguments:
    landmarks_px-- landmarks in frame space returned by landmark_predictor to be reflected
    points_z    -- z-distance in mm of points_px in metric space (Crutch: assumed they are on a xy-parallel plane)
    viewpoint_px-- dlib.point of viewpoint in frame space
    viewpoint_z -- z-distance in mm of viewpoint_px  
    '''
    points_onscreen = {}
    points_mm = {}
    #viewpoint_mm = frame_to_metric(viewpoint_px.x, viewpoint_px.y, viewpoint_z)
    viewpoint_mm = (-161, 108, 1419)
    for key in landmarks_px.keys():
        p = landmarks_px[key]
        p_mm = frame_to_metric(p[0], p[1], points_z)
        points_mm[key] = p_mm

        p_onscreen = screen_refection(p_mm, viewpoint_mm)
        points_onscreen[key] = p_onscreen
    
    return (points_onscreen, points_mm)


def landmark_reflection_from_frame(rgb_frame, depth_frame, debug=False):
    # Isolate interest quad
    (interest_x, interest_y, interest_w, interest_h) = interest_quad
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    gray_frame_interest = cv2.cvtColor(rgb_frame[interest_y:interest_y+interest_h, interest_x:interest_x+interest_w], cv2.COLOR_BGR2GRAY)
    if debug:
        quad_p1 = (interest_x, interest_y)
        quad_p2 = (interest_x + interest_w, interest_y + interest_h)
        cv2.rectangle(rgb_frame, quad_p1, quad_p2, (255, 255, 255), 5)
        rgb_frame[quad_p1[1]:quad_p2[1], quad_p1[0]:quad_p2[0]] = cv2.cvtColor(gray_frame_interest, cv2.COLOR_GRAY2RGB)
    
    faces = face_detector(image=gray_frame_interest)
    
    # Draw a green rectangle around the faces
    for face_rect in faces:
        # Account for interest quad crop
        (x, y, w, h) = rect_to_bb(face_rect)
        x += interest_x
        y += interest_y

        if debug:
            cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Find viewpoint inside face
        landmarks = landmark_predictor(image=gray, box=dlib.rectangle(x, y, x+w, y+h))
        viewpoint_px = dlib.point((landmarks.part(LANDMARK_IDX['r_eye_r']) + landmarks.part(LANDMARK_IDX['r_eye_l'])) / 2)

        if debug:
            # Draw all the landmarks
            for n in range(0, LANDMARK_LEN):
                l_x = landmarks.part(n).x
                l_y = landmarks.part(n).y

                cv2.circle(img=rgb_frame, center=(l_x, l_y), radius=1, color=(0, 255, 0), thickness=0)

        # Find middle of face. Use depth from there 
        midface_px = (int(x + w / 2), int(y + h / 2))
        midface_z = depth_frame[midface_px[1], midface_px[0]]
        
        if debug:
            cv2.circle(rgb_frame, (viewpoint_px.x, viewpoint_px.y), 5, (255,255,255))
        
        reduced_landmarks = {}
        for key, idx in LANDMARK_IDX.items():
            reduced_landmarks[key] = (landmarks.part(idx).x, landmarks.part(idx).y)
        
        return reflect_landmarks(reduced_landmarks, midface_z, viewpoint_px, midface_z)
    return (None, None)

# Normalizes xy-point in frame (center becomes (0, 0))
def normalize_frame_point(p):
    x_norm = p[0] / image_w - 0.5
    y_norm = p[1] / image_h - 0.5
    return (x_norm, y_norm)

sin_fov_xo2 = math.sin(math.radians(fov_x/2))
sin_fov_yo2 = math.sin(math.radians(fov_y/2))
def get_frame_w(z):
    return 2 * sin_fov_xo2 * z

def get_frame_h(z):
    return 2 * sin_fov_yo2 * z

#Returns median of 5px around point
Z_SAMPLE_R = 5
def sample_z(frame, x, y):
    return np.median(frame[x-Z_SAMPLE_R:x+Z_SAMPLE_R, y-Z_SAMPLE_R:y+Z_SAMPLE_R])


def place_nose(img, nose, landmarks):
    w = int((landmarks['nose_l'][0] - landmarks['nose_r'][0]))
    h = int((w * nose.shape[0] / nose.shape[1]))
    w = int(w*1.8)
    h = int(h*1.8)
    y = int(landmarks['nose'][1] - h/2)
    x = int(landmarks['nose'][0] - w/2)
    if (w > 0 and h > 0):
        nose = cv2.resize(nose, (w, h))
        overlay_img(img, nose, x, y)
def overlay_img(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x < 0:
        if -x > w:
            return background
        w = w + x
        x = 0
        overlay = overlay[:, :w]
    if y < 0:
        if -y > h:
            return background
        h = h + y
        y = 0
        overlay = overlay[:h]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]


    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0
    background[y:y+h, x:x+w] = cv2.add(background[y:y+h, x:x+w],overlay)

    return background

# Returns (x, y, w, h)
IQ_SCALE = 1.0
def get_interest_quad(): 
    z_min = op_distance[0]
    z_max = op_distance[1]
    quad = (0, 0, 0, 0)

    frame_w_min = get_frame_w(z_min)
    frame_h_min = get_frame_h(z_min)
    frame_w_max = get_frame_w(z_max)
    frame_h_max = get_frame_h(z_max)
    
    # Screen coordinates in metric space centered around webcam (+y = up, +x = left looking from webcam)
    screen_x_mm = webcam_x - screen_mm[0] / 2 
    screen_y_mm = -cam_screen_offset_y

    x_min = image_w * (screen_x_mm + frame_w_min / 2) / frame_w_min 
    y_min = image_h * (-screen_y_mm + frame_h_min / 2) / frame_h_min
    w_min = image_w * screen_mm[0] / frame_w_min
    h_min = image_h * screen_mm[1] / frame_h_min

    x_max = image_w * (screen_x_mm + frame_w_max / 2) / frame_w_max 
    y_max = image_h * (-screen_y_mm + frame_h_max / 2) / frame_h_max
    w_max = image_w * screen_mm[0] / frame_w_max
    h_max = image_h * screen_mm[1] / frame_h_max

    x = min(x_min, x_max)  
    y = min(y_min, y_max)
    w = max(w_min, w_max)
    h = max(h_min, h_max)
    x = int(x - w * (IQ_SCALE - 1) / 2)
    y = int(y - h * (IQ_SCALE - 1) / 2)
    w = int(w * IQ_SCALE)
    h = int(h * IQ_SCALE)
    
    return (max(x, 0), max(y, 0), w, h)

# q1, q2 - Quads in format (x, y, w, h)
def point_in_quad(p, q):
    (xq, yq, wq, hq) = q
    (xp, yp) = p
    return xp > xq and xp < xq + wq and yp > yq and yp < yq + hq
 
click_coords = None
mouse_coords = None
calibrating = None
def mouse_cb(event, x, y, flags, param):
    global click_coords, mouse_coords
    if (event == cv2.EVENT_LBUTTONDOWN and calibrating):
        click_coords = dlib.point(x, y)
    elif (event == cv2.EVENT_MOUSEMOVE):
        mouse_coords = dlib.point(x, y)

def calibrate_offsets():
    global calibrating, px_offset_x, px_offset_y
    t0 = -1
    
    # Create prompt images
    prompt_face = np.zeros((screen_res[1], screen_res[0], 3), np.uint8)
    cv2.putText(prompt_face, 'Stand in front of the mirror (1-3m away)', (100, screen_res[1] - 100), 
        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255))
    prompt_main = np.zeros((screen_res[1], screen_res[0], 3), np.uint8)
    cv2.putText(prompt_main, 'Click on the tip of your nose', (100, screen_res[1] - 100), 
        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255))
    
    while 1:
        rgb_frame = get_rgb()
        depth_frame = get_depth()

        (landmarks_px, landmarks_mm) = landmark_reflection_from_frame(rgb_frame, depth_frame, debug)

        if landmarks_px and landmarks_mm:
            calibrating = True
            nose = landmarks_px[LANDMARK_IDX['nose']]
          
            image = np.copy(prompt_main)
            if (debug):
                cv2.circle(image, nose, 5, (0, 255, 0), 2)
            
            if (click_coords):
                print('Click detected in loop')
                # Calibrate
                # Offset is assumed nose position - actual eye position 
                px_offset_x = click_coords.x - nose[0]
                px_offset_y = click_coords.y - nose[1]
                
                calibrating = False
                break
            cv2.imshow('window', image)
            
        else: 
            calibrating = False
            # No Faces
            cv2.imshow('window', prompt_face)
        if cv2.waitKey(1) & 0xFF == ord('q'):   
            break


 
# Function to get RGB image from kinect
def get_rgb():
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return array
 
# Function to get depth image from kinect
def get_depth(pretty=False):
    array, _ = freenect.sync_get_depth(0, freenect.DEPTH_11BIT if pretty else freenect.DEPTH_REGISTERED)
    if pretty:
        array = array.astype(np.uint8)
    return array

''' 
    PREP & INIT
'''
# Calculate interest quad in frame space, from which a reflection would land inside the monitor
# Only video from this quad is considered
interest_quad = get_interest_quad()

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
LANDMARK_LEN = 68
LANDMARK_IDX = {
    'nose': 33,
    'r_eye_r': 45,
    'r_eye_l': 42,
    'nose_l': 31,
    'nose_r': 35
}

cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.moveWindow("window", 2000, 0)
cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback("window", mouse_cb)

'''
# Tilt to 0deg
cam_angle = 0
ctx = freenect.init()
dev = freenect.open_device(ctx, 0)
freenect.set_tilt_degs(dev, cam_angle)
print(freenect.get_tilt_degs(freenect.get_tilt_state(dev)))
freenect.close_device(dev)'''

dog_lear = cv2.imread('./dogfilter_lear.png')
dog_rear = cv2.imread('./dogfilter_rear.png')
dog_nose = cv2.imread('./dogfilter_nose.png')

#calibrate_offsets()

while 1:
    rgb_frame = get_rgb()
    depth_frame = get_depth()
    image = np.zeros((screen_res[1], screen_res[0], 3), np.uint8)
    cv2.circle(image, (screen_res[0] - 40, screen_res[1] - 20), 10, (0, 255, 0), cv2.FILLED)

    (landmarks_px, landmarks_mm) = landmark_reflection_from_frame(rgb_frame, depth_frame, debug=True)
        
    if landmarks_px and landmarks_mm:
        nose = landmarks_mm['nose']
        place_nose(image, dog_nose, landmarks_px)
        if debug:
            cv2.putText(image, "Nose at ({:.1f}mm, {:.1f}mm)".format(nose[0], nose[1]), (20
                    , 20), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 0))

    if debug:
        # Overlay debug camera image
        debug_rgb = cv2.resize(rgb_frame, (debug_w, debug_h))
        image[-debug_h:, -debug_w:] = debug_rgb

    cv2.imshow('window', image)
    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):   
        break
    '''
#Show cam stream and provide metric xyz of point at which mouse is pointing
while 1:
    rgb_frame = get_rgb()
    depth_frame = get_depth()
    
    
    pretty_depth = depth_frame / 15000 * 255
    pretty_depth = cv2.cvtColor(pretty_depth.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    #rgb_frame = cv2.addWeighted(rgb_frame, 0.5, pretty_depth, 0.5, 0)
    
    
    if (mouse_coords and mouse_coords.x > 0 and mouse_coords.y > 0):
        x_px = mouse_coords.x
        y_px = mouse_coords.y
        z = sample_z(depth_frame, x_px, y_px)
        (x_mm, y_mm, z_mm) = frame_to_metric(mouse_coords.x, mouse_coords.y, z)

        cv2.circle(pretty_depth, (x_px, y_px), 5, (255, 0, 0)) #draw mouse
        cv2.putText(pretty_depth, "({:.1f}mm, {:.1f}mm, {:.1f}mm)".format(x_mm, y_mm, z_mm), (20
                    , 20), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 0))
    
    cv2.imshow('window', pretty_depth)

    
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):   
        break
'''

raise freenect.Kill
video_capture.release()
cv2.destroyAllWindows()

