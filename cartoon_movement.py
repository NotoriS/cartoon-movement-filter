import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

SPEED_LINE_APPEARANCE_DELAY = 0.2
SPEED_LINE_LIFETIME = 0.5
TRACKING_POINT_RESELECTION_DELAY = 0.4

# Parameters for lucas kanade optical flow
lk_params = dict(
    winSize  = (21, 21),
    maxLevel = 3,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Fetch input filename and create output filename
input_video_name = sys.argv[1]
split_video_name = input_video_name.split('.')
if split_video_name[-1] != 'mp4':
    print('Error: this filter only accepts .mp4 files as input.')
    sys.exit()
split_video_name[-2] = split_video_name[-2] + '_cartoon_movement'
output_video_name = '.'.join(split_video_name)

# Fetch relevant information about the input video
cap = cv2.VideoCapture(input_video_name)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

os.makedirs('out', exist_ok=True)
os.chdir('./out/')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_name, fourcc, fps, (frame_w, frame_h), isColor=True)

bg_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

iteration = -1
while cap.isOpened():
    iteration = iteration + 1

    ret, frame = cap.read()
    if not ret:
        break

    if iteration == 0:
        mask = np.zeros_like(frame)

    # Apply adaptive background subtraction to original frame 
    # and blur it to expand the movement region.
    if iteration > 0:
        old_no_bg_frame = no_bg_frame
    no_bg_frame = bg_subtractor.apply(frame)
    no_bg_frame_blurred = cv2.GaussianBlur(no_bg_frame, (7, 7), 5)

    # Blur the oringal frame in order to acheive better results from edge detection
    blurred_frame = cv2.GaussianBlur(frame, (3, 3), 1)

    # Detect edges to determine what areas to emit speed lines from
    edge_frame = cv2.Canny(blurred_frame, 60, 60)

    # Narrow down the detected edges to the ones that are moving
    moving_edges = np.zeros_like(edge_frame)
    moving_edges[no_bg_frame_blurred > 0] = edge_frame[no_bg_frame_blurred > 0]

    if iteration % round(fps * TRACKING_POINT_RESELECTION_DELAY) == 0 or len(p0) == 0:
        # Reselect lucas-kanade tracking points to account for new on-screen objects
        p0 = np.argwhere(moving_edges > 0)
        p0 = p0[:, np.newaxis, :].astype(np.float32)
    else:
        # Continue using previously selected points
        p0 = good_new.reshape(-1, 1, 2)

    # Skip to next iteration if there are no good tracking points yet
    if len(p0) == 0 or iteration == 0:
        out.write(frame)
        continue

    # Track edges using the lucas-kanade method
    p1, st, _ = cv2.calcOpticalFlowPyrLK(old_no_bg_frame, no_bg_frame, p0, None, **lk_params)

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    for new, old in zip(good_new, good_old):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (200, 200, 200), 1)

    out.write(cv2.add(mask, frame))

cap.release()
out.release()
cv2.destroyAllWindows()

os.chdir('../')