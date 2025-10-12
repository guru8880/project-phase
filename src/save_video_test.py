import cv2
import numpy as np
import os

video_folder = "data/test_videos"
video_test_list = []

for video_file in os.listdir(video_folder):
    if not video_file.endswith(".mp4"):
        continue
    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (150, 150))
        frame = frame / 255.0
        video_test_list.append(frame)
    cap.release()

video_test = np.array(video_test_list)
os.makedirs("data/test", exist_ok=True)
np.save("data/test/video_test.npy", video_test)
print("✅ video_test.npy created successfully!")
