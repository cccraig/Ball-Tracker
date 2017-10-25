from collections import deque
from threading import Thread
from Queue import Queue
import numpy as np
import cv2


class Follow():

    def __init__(self):

        # Capture the webcam
        self.camera = cv2.VideoCapture(-1)
        self.camera.set(3,1920)
        self.camera.set(4,1080)

        # Ball (BGR)
        self.bl = (35, 75, 50)
        self.bu = (85, 155, 255)

        # Buffer for tracking
        self.buff = 15

        # Minimum radius
        self.min_radius = 10
        self.max_radius = 100

        # Width of circle border
        self.width = 10

        # Points for our tracker
        self.bpts = deque(maxlen=self.buff)

        # Circle color
        self.circle = (0, 127, 0)

        # Line trail
        self.line = (0, 89, 217)

        # Size of roi square
        self.roi_size = 30

        # Pause drawing
        self.pause = 0

        # Pause drawing
        self.halt = 0

        # Show mask
        self.show_mask = 0

        # Morphology Kernels
        self.hort_line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,5))
        self.vert_line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,1))
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))




    def calibrate(self):

        # Roi size
        r = self.roi_size

        # Calibration count
        i = 4

        while not self.pause:

            # Grab the current web frame
            frame = self.camera.read()[1]

            # Deep copy
            img = frame.copy()

            # Size of frame
            y, x, _ = np.shape(img)

            # Image locations
            xx, yy = int(x - x/2), int(y - y/2)

            # Draw square
            cv2.rectangle(img, (xx-r, yy-r), (xx+r, yy+r), (0, 0, 255), 2)

            # Flip frame to help with movement
            img = cv2.flip(img, 1)

            # Show the frame
            cv2.imshow('Frame', img)

            # exit on escape
            k = cv2.waitKey(5) & 0xFF

            if k == 27:
                self.halt = 1
                break

            # Capture color on c
            if k == 99 or k == 32:
                roi = frame[yy-r:yy+r,xx-r:xx+r]
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                self.pause = 1

        image_mean = np.ceil(hsv.mean(axis=(0,1)))
        image_std = np.ceil(hsv.std(axis=(0,1)))

        self.bl = image_mean - image_std * 2
        self.bu = image_mean + image_std * 2
        self.pause = 0

        self.start_thread()
        self.capture()




    def filter_objects(self, cnts):

        c = max(cnts, key=cv2.contourArea)

        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if radius > self.min_radius and radius < self.max_radius:
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            return ((int(x), int(y)), int(radius), center)

        return ((0, 0), 0, None)




    def ball(self, hsv, frame):

        # Construct a mask
        mask = cv2.inRange(hsv, self.bl, self.bu)
        mask = cv2.erode(mask, self.kernel, iterations=2)
        mask = cv2.dilate(mask, self.kernel, iterations=2)

        # Find contours in the mask and initialize the current (x,y) center of ball
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        # Short name for points
        pts = self.bpts

        # Continue only if at least one contour was found
        if len(cnts) > 0:

            ((x, y), radius, center) = self.filter_objects(cnts)

            # Proceed only if we detected an appropriate size blob
            if radius is not 0:
                cv2.circle(frame, (x, y), radius, self.circle, self.width)
                cv2.circle(frame, center, 5, self.line, -1)

                # Update the points deque
                if center is not None:
                    pts.appendleft(center)

                # loop over the set of tracked points
                for i in xrange(1, len(pts)):
             
                    # otherwise, compute the thickness of the line and draw the connecting lines
                    thickness = int(np.sqrt(self.buff / float(i + 1)) * 5)
                    cv2.line(frame, pts[i - 1], pts[i], self.line, thickness)
     
            else:
                # If no circle, reset the line tracker
                pts = deque(maxlen=self.buff)

        self.bpts = pts 

        return cv2.flip(frame,1), mask



    # Use a thread for speed
    def start_thread(self):

        # Stop flag
        self.halt = 0

        # Frame queue
        self.f_queue = Queue()

        # Initialize thread
        self.t = Thread(target=self.update)

        # Make daemon (dies automatically)
        self.t.daemon = True

        # Start the thread
        self.t.start()



    # Thread worker function
    def update(self):

        while not self.halt:

            # Grab the current web frame
            frame = self.camera.read()[1]

            # Put it in the queue
            self.f_queue.put(frame)





    def capture(self):

        # Loop untill we stop
        while not self.halt:

            if not self.f_queue.empty():

                # Grab the current web frame
                frame = self.f_queue.get()

                # Convert to HSV colorspace
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # Look for balls
                frame, mask = self.ball(hsv, frame)

                # show the frame to our screen
                cv2.imshow("Frame", frame)

                if self.show_mask:
        			cv2.imshow('Mask', mask)

                # exit on escape
                k = cv2.waitKey(5) & 0xFF

                if k == 27:
                    self.halt = 1
                    break

                if k == 109:
                	if self.show_mask:
                		self.show_mask = 0
                		cv2.destroyWindow('Mask')
                	else:
                		self.show_mask = 1

                if k == 114:
                    self.halt = 1

        if k == 114:
            cv2.destroyWindow('Mask')
            self.calibrate();

        else:
            # Release the Camera
            self.camera.release()

            # Destroy all windows
            cv2.destroyAllWindows()

tracker = Follow()
tracker.calibrate()
# tracker.start_thread()
# tracker.capture()