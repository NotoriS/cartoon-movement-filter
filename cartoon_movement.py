import sys
import os
import cv2
import numpy as np

SPEED_LINE_APPEARANCE_DELAY = 0.2
SPEED_LINE_LIFETIME = 0.5
TRACKING_POINT_RESELECTION_DELAY = 0.4

BG_SUBTRACTION_MEANS = 5

# Parameters for corner detection
cd_params = dict( 
    maxCorners = 200,
    qualityLevel = 0.01,
    minDistance = 10,
    blockSize = 7
)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(
    winSize  = (21, 21),
    maxLevel = 4,
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

recent_gray_frames = []
iteration = -1
while cap.isOpened():
    iteration = iteration + 1

    ret, frame = cap.read()
    if not ret:
        break

    if iteration == 0:
        mask = np.zeros_like(frame)

    # Apply adaptive background subtraction to original frame 
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    recent_gray_frames.append(gray_frame)
    if len(recent_gray_frames) > BG_SUBTRACTION_MEANS:
        recent_gray_frames.pop(0)
    else:
        out.write(frame)
        continue
    median = np.median(np.stack(recent_gray_frames, axis=0), axis=0).astype(np.uint8)
    if iteration > BG_SUBTRACTION_MEANS:
        old_no_bg_frame = no_bg_frame
    no_bg_frame = cv2.absdiff(gray_frame, median)
    if (iteration < BG_SUBTRACTION_MEANS + 1):
        out.write(frame)
        continue

    if (iteration - BG_SUBTRACTION_MEANS - 1) % round(fps * TRACKING_POINT_RESELECTION_DELAY) == 0:
        mask = np.zeros_like(frame)
        # Apply corner detection in order to find features to track using the Lucas-Kanade method
        p0 = cv2.goodFeaturesToTrack(old_no_bg_frame, mask = None, **cd_params)
    else:
        # Continue using previously selected points
        p0 = good_new.reshape(-1, 1, 2)

    # Skip to next iteration if there are no good tracking points yet
    if len(p0) == 0:
        out.write(frame)
        continue

    # Calculate the optical flow the Lucas-Kanade method
    p1, st, _ = cv2.calcOpticalFlowPyrLK(old_no_bg_frame, no_bg_frame, p0, None, **lk_params)

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    for new, old in zip(good_new, good_old):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (150, 150, 150), 1)

    out.write(cv2.add(mask, frame))

cap.release()
out.release()
cv2.destroyAllWindows()

os.chdir('../')