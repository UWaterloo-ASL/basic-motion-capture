# USAGE
# python motion_detector.py
# python motion_detector.py --video videos/example_01.mp4

# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-v", "--video", default='./videos/example_01.mp4', help="path to the video file")
#ap.add_argument("-v", "--video", default='./videos/example_02.mp4', help="path to the video file")
#ap.add_argument("-v", "--video", default='./videos/LAS_ROM2.mov', help="path to the video file")
ap.add_argument("-v", "--video", default='./videos/LAS_ROM.mov', help="path to the video file")
#ap.add_argument("-v", "--video", help="path to the video file")
#ap.add_argument("-v", "--video", default='./videos/vtest.avi', help="path to the video file")

ap.add_argument("-min-a", "--min-area", type=int, default=500, help="minimum area size")
ap.add_argument("-max-a", "--max-area", type=int, default=5000, help="maximum area size")

ap.add_argument("-bg-sub", "--bg-subtractor", default='fgbg_mog',
                help="1. absdiff; 2. fgbg_mog; 3. fgbg_mog2;4. fgbg_gmg")

ap.add_argument("-fwidth", "--fwidth", default=500, help="set frame width")

args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    camera = cv2.VideoCapture(0)
    time.sleep(0.25)
# otherwise, we are reading from a video file
else:
    camera = cv2.VideoCapture(args["video"])

# prepare to save video
(grabbed, frame) = camera.read()
frame = imutils.resize(frame, width=args['fwidth'])
fshape = frame.shape
fheight = fshape[0]
fwidth = fshape[1]
print("Frame width:{}, Frame height:{}.".format(fwidth , fheight))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_camera_frame = cv2.VideoWriter('{}_camera_frame.avi'.format(args['bg_subtractor']),fourcc, 20.0, (fwidth,fheight))
out_fbmask = cv2.VideoWriter('{}_fbmask.avi'.format(args['bg_subtractor']),fourcc, 20.0, (fwidth,fheight),isColor=False)
out_dilated_thresh = cv2.VideoWriter('{}_dilated_thresh.avi'.format(args['bg_subtractor']),fourcc, 20.0, (fwidth,fheight),isColor=False)

########################################################
#        Define Different Background Subtractor        #
########################################################
# 1. compute the absolute difference between the current and first frame
#    do not need to initialize subtractor
    
# 2. BackgroundSubtractorMOG
fgbg_mog = cv2.bgsegm.createBackgroundSubtractorMOG()

# 3. BackgroundSubtractorMOG2 
fgbg_mog2 = cv2.createBackgroundSubtractorMOG2()

# 4. BackgroundSubtractorGMG
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg_gmg = cv2.bgsegm.createBackgroundSubtractorGMG()

# initialize the first frame in the video stream
# only needed when use absolute difference between the current and first frame
firstFrame_im= cv2.imread('./videos/first_frame_LAS_ROM.png')
firstFrame_im = imutils.resize(firstFrame_im, width=args['fwidth'])
firstFrame = cv2.cvtColor(firstFrame_im, cv2.COLOR_BGR2GRAY)
firstFrame = cv2.GaussianBlur(firstFrame, (21, 21), 0)

firstFrame = None

# loop over the frames of the video
while True:
    # grab the current frame and initialize the occupied/unoccupied
    # text
    (grabbed, frame) = camera.read()

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not grabbed:
        break

    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=args['fwidth'])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)

    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue
    
    if args['bg_subtractor'] == 'absdiff':
        # 1. compute the absolute difference between the current frame and
        # first frame
        fgmask = cv2.absdiff(firstFrame, gray)
    elif args['bg_subtractor'] == 'fgbg_mog':
        # 2. BackgroundSubtractorMOG
        fgmask = fgbg_mog.apply(gray)
    elif args['bg_subtractor'] == 'fgbg_mog2':
        # 3. BackgroundSubtractorMOG2
        fgmask = fgbg_mog2.apply(gray)
    elif args['bg_subtractor'] == 'fgbg_gmg':
        # 4. BackgroundSubtractorGMG
        fgmask = fgbg_gmg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    else:
        ValueError("Please choose a right background subtractor!")
    
    ret, thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    occupancy = 0
    # loop over the contours
    for c in contours:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]: #or cv2.contourArea(c) < args["max_area"]:
            #print("contourArea size:{}".format(cv2.contourArea(c)))
            continue
        occupancy += 1
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(fgmask, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.rectangle(thresh, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # draw the text and timestamp on the frame, thresh, fgmask
    cv2.putText(frame, "{} Current Occupancy: {}".format(args['bg_subtractor'],occupancy), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    
    cv2.putText(fgmask, "{} Current Occupancy: {}".format(args['bg_subtractor'],occupancy), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(fgmask, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    
    cv2.putText(thresh, "{} Current Occupancy: {}".format(args['bg_subtractor'],occupancy), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(thresh, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # show the frame and record if the user presses a key
    cv2.imshow("{}: Camera Frame".format(args['bg_subtractor']), frame)
    cv2.imshow("{}: For-background Mask".format(args['bg_subtractor']), fgmask)
    cv2.imshow("{}: Dilated Thresh".format(args['bg_subtractor']), thresh)
    # save video
    out_camera_frame.write(frame)
    out_fbmask.write(fgmask)
    out_dilated_thresh.write(thresh)
    
    
    
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()