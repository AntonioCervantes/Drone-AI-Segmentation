""" This program detects a ArUco Tag on both images and videos, based on the compatible dictionaries. 
    Please modify user arguments and/or global constants to modify the ArUco tag generated.

    Author:         Christian Pedrigal, pedrigalchristian@gmail.com
    Last modified:  3/12/2022
"""

import time
import cv2
import imutils
from imutils.video import VideoStream
import numpy as np
from typing import Tuple

from numpy import ndarray

from aruco_dict import ARUCO_DICT
from create_aruco_markers import tag_dict

# Global Constants
DEBUG: bool      = False
VIDEO: bool      = True
VIDEO_WIDTH: int = 1000

# User Arguments
frame_path: str =  r"C:\Users\pedri\Pictures\Camera Roll\WIN_20220406_06_29_19_Pro.jpg"

# User Defined Functions
def _detect_aruco(frame: np.ndarray) -> Tuple: # Returns a tuple of (tuple, numpy.ndarray)

    # Creating default center coordinates
    cX = None
    cY = None

    # Detect Aruco Marker
    arctype: cv2.aruco_Dictionary               = cv2.aruco.Dictionary_get(ARUCO_DICT[tag_dict])
    aruco_params: cv2.aruco_DetectorParameters  = cv2.aruco.DetectorParameters_create()
    (corner_list, id_list, rejected_list)       = cv2.aruco.detectMarkers(frame, arctype, parameters = aruco_params)

    # Annotate Detections on Frame
    if len(corner_list) > 0:
        if DEBUG: print("Marker Detected!")
        id_list = id_list.flatten() # Flatten to 1 row

        # Reshape corners into readable list
        for (obj_corners, obj_id) in zip(corner_list, id_list): # For each corner in all corners
            obj_corners = obj_corners.reshape((4,2)) # Shapes numpy array into list of x-y coordinates
            (topLeft, topRight, botRight, botLeft) = obj_corners

            # Cast as integers (Pixels are only Integers)
            topLeft  = (int(topLeft[0]), int(topLeft[1]))
            topRight = (int(topRight[0]), int(topRight[1]))
            botRight = (int(botRight[0]), int(botRight[1]))
            botLeft  = (int(botLeft[0]), int(botLeft[1]))

            # Compute center coordinates of ID
            cX = int( (topLeft[0] + botRight[0]) / 2.0 )
            cY = int( (topLeft[1] + botRight[1]) / 2.0 )
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

            # Draw Coordinates on Text
            text = f"(Center: ({cX}, {cY}), ID: {obj_id})"
            frame = cv2.putText(img = frame, 
                                text = text, 
                                org = (topLeft[0], topLeft[1] - 15),
                                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale = 0.5, 
                                color = (0, 0, 255), 
                                thickness = 2)
    if DEBUG: 
        print(type((cX, cY)))
        print(type(frame))

    return ((cX, cY), frame)

def main(frame_path: str) -> None:
    """ This main program works with single images specified from the full_path specified from the user.

        Parameters:
        frame_path: a raw string of the full path to the image
    """
    # Import Frame
    frame = cv2.imread(frame_path)
    frame = imutils.resize(frame, width = 600)

    (_, frame) = _detect_aruco(frame)

    cv2.imshow('Frame', frame)
    cv2.waitKey(0)

def main2() -> None:
    """ This main program utilizes the webcam to detect ArUco tags in real-time.

        Parameters:
        None
    """

    # Create a video object
    vs = imutils.video.VideoStream(src = 0).start()
    time.sleep(2.0)
    if DEBUG: print("Starting up the webcam...")

    while True:
        # Read a single frame
        frame = vs.read() 
        frame = imutils.resize(frame, width = VIDEO_WIDTH)
    
        (_, output) = _detect_aruco(frame)
        cv2.imshow('Frame', output)
        key = cv2.waitKey(1)

        if key == ord('q'):
            cv2.destroyAllWindows()
            vs.stop()
            break

# Main Program
if __name__ == "__main__":
    if VIDEO:
        main2()
    if not VIDEO:
        main(frame_path)