import sys
import os
import cv2
import numpy as np

SPEED_LINE_APPEARANCE_DELAY = 0.1
SPEED_LINE_LIFETIME = 0.4
TRACKING_POINT_RESELECTION_DELAY = 0.3

BG_SUBTRACTION_FRAME_COUNT = 5

# Fetch input filename and create output filename
input_video_name = sys.argv[1]
split_video_name = input_video_name.split('.')
if split_video_name[-1] != 'mp4':
    print('Error: this filter only accepts .mp4 files as input.')
    sys.exit()
split_video_name[-2] = split_video_name[-2] + '_cartoon_movement'
output_video_name = '.'.join(split_video_name)
output_video_name = output_video_name.replace('\\', '/')
output_video_name = output_video_name.split('/')[-1]

# Fetch relevant information about the input video
cap = cv2.VideoCapture(input_video_name)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Define parameters for Shi-Tomasi corner detection
cd_params = dict( 
    maxCorners = 200,
    qualityLevel = 0.05,
    minDistance = max(frame_w, frame_h) // 100,
    blockSize = 20
)

# Define parameters for Lucas-Kanade optical flow
lk_params = dict(
    winSize  = (frame_w // 30, frame_h // 30),
    maxLevel = 5,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)
)

# Define minimun and maximum tracked movements per frame
minimum_movement = (max(frame_w, frame_h) / 1000) - 1
maximum_movement = max(frame_w, frame_h) / 15

# Calculate values to later add delay and lifetime to the speed lines
new_masks_skipped = round(fps * SPEED_LINE_APPEARANCE_DELAY)
masks_drawn_per_frame = round(fps * SPEED_LINE_LIFETIME)

os.makedirs('out', exist_ok=True)
os.chdir('./out/')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_name, fourcc, fps, (frame_w, frame_h), isColor=True)

recent_gray_frames = []
queued_line_masks = []
iteration = -1
while cap.isOpened():
    iteration = iteration + 1

    ret, frame = cap.read()
    if not ret:
        break

    # Apply adaptive background subtraction to original frame 
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    recent_gray_frames.append(gray_frame)
    if len(recent_gray_frames) > BG_SUBTRACTION_FRAME_COUNT:
        recent_gray_frames.pop(0)
    else:
        out.write(frame)
        continue
    median = np.median(np.stack(recent_gray_frames, axis=0), axis=0).astype(np.uint8)
    if iteration > BG_SUBTRACTION_FRAME_COUNT:
        old_no_bg_frame = no_bg_frame
    no_bg_frame = cv2.absdiff(gray_frame, median)
    if (iteration < BG_SUBTRACTION_FRAME_COUNT + 1):
        out.write(frame)
        continue

    if (iteration - BG_SUBTRACTION_FRAME_COUNT - 1) % round(fps * TRACKING_POINT_RESELECTION_DELAY) == 0:
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

    # Draw the lines created by the movement this frame on a mask
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    line_mask = np.zeros_like(no_bg_frame)
    for new, old in zip(good_new, good_old):
        a, b = new.ravel()
        c, d = old.ravel()
        distance = np.sqrt((a - c)**2 + (b - d)**2)
        if minimum_movement <= distance <= maximum_movement:
            line_mask = cv2.line(line_mask, (int(a), int(b)), (int(c), int(d)), 1, 1)

    # Add the lines calculated this frame to the list of lines to be drawn
    queued_line_masks.insert(0, line_mask)
    if len(queued_line_masks) > new_masks_skipped + masks_drawn_per_frame:
        queued_line_masks.pop()

    # Draw all of the currently queued line masks except for the newest few
    cumulative_mask = np.zeros_like(line_mask)
    for i in range(new_masks_skipped, new_masks_skipped + masks_drawn_per_frame):
        if i >= len(queued_line_masks):
            break
        cumulative_mask = cv2.add(cumulative_mask, queued_line_masks[i])

    # Convert the cumulative mask to a colored image and convolve it with 
    # a gaussian kernel to soften and thicken the lines
    colored_mask = np.zeros_like(frame)
    colored_mask[cumulative_mask > 0] = (255, 255, 255)
    colored_mask = cv2.GaussianBlur(colored_mask, (7, 7), 0.8)
    frame = cv2.add(colored_mask, frame)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

os.chdir('../')