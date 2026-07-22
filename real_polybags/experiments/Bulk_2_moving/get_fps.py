# import cv2

# video_path = "lucid_1280x720_20260528_100129.avi"
# cap = cv2.VideoCapture(video_path)

# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# duration = frame_count / fps if fps > 0 else 0

# print(f"FPS: {fps}")
# print(f"Total Frames: {frame_count}")
# print(f"Duration: {duration:.2f} seconds")

# cap.release()

import cv2
import os

# Settings
video_path = "lucid_1280x720_20260528_100129.avi"
output_folder = "frames"
frame_interval = 1  # Extract every N frames (1 = every frame)

# Create output folder
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"FPS: {fps}")
print(f"Total Frames: {total_frames}")

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"✅ Done! Saved {saved_count} images to '{output_folder}' folder")