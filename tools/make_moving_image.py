import cv2
from pathlib import Path
import argparse
import numpy as np


argparser = argparse.ArgumentParser()
argparser.add_argument("--file", type=str, default="../data/input/md/DJI_0706.MP4", help="input video")
argparser.add_argument("--from_frame", type=int, default=250, help="from which frame to save")
argparser.add_argument("--frame_num", type=int, default=100, help="total frame number")
argparser.add_argument("--out_file", type=str, default="out.bin", help="output file")
argparser.add_argument("--img_type", type=int, default=1, help="0: gray, 1: yuv420sp")

flags = argparser.parse_args()

roi_upleft = [1682, 1182]
roi_wh = [1116, 820]

cap = cv2.VideoCapture(flags.file)

def BGR2yuv420sp(img, imgH, imgW):
    img = cv2.resize(img, (imgW, imgH))

    # BGR to YUV420sp
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
    h, w = yuv.shape
    h_plane = h // 3
    yuv_sp = yuv.copy()
    uv_sp = yuv_sp[h_plane * 2:].reshape(2, h_plane // 2, w).transpose(1, 2, 0).reshape(h_plane, -1)
    yuv_sp[h_plane * 2:] = uv_sp
    return yuv_sp

def cut_roi(img, roi_upleft, roi_wh, offset=[0, 0]):
    x0, y0 = np.array(roi_upleft) + offset
    x1, y1 = x0 + roi_wh[0], y0 + roi_wh[1]
    roi = img[y0:y1, x0:x1]
    return roi


def assign_roi(roi, upleft):
    empty_img = np.zeros((2160, 3840, 3), dtype=np.uint8)
    x0, y0 = upleft
    h, w = roi.shape[:2]
    x1, y1 = x0 + w, y0 + h
    empty_img[y0:y1, x0:x1] = roi
    return empty_img


v = np.array([3, 0])  # vx, vy
offset = np.array([0, 0])
if not cap.isOpened():
    print("Error: Cannot open video!")
else:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"total number of frames: {total_frames}")

    if flags.from_frame >= total_frames:
        print(f"Error: start frame: {flags.from_frame} should less than total_frame: {total_frames}")
    else:
        frame_id = 0
        end_frame = min(flags.from_frame + flags.frame_num, total_frames)
        frames = []
        while True:
            ret, frame = cap.read()
            frame_id += 1
            if not ret:
                continue
            if flags.from_frame <= frame_id  < end_frame:

                if flags.img_type == 0:
                    offset += v
                    roi = cut_roi(frame, roi_upleft, roi_wh, offset)
                    frame = assign_roi(roi, roi_upleft)
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    h, w = gray_frame.shape[:2]
                    if h != 1080 or w != 1920:
                        gray_frame = cv2.resize(gray_frame, (1920, 1080))

                    frames.append(gray_frame)
                    cv2.imwrite(f"img_{frame_id:04d}.jpg", gray_frame)
                elif flags.img_type == 1:
                    offset += v
                    roi = cut_roi(frame, roi_upleft, roi_wh, offset)
                    frame = assign_roi(roi, roi_upleft)
                    cv2.imwrite(f"img_{frame_id:04d}.jpg", frame)
                    yuv = BGR2yuv420sp(frame, 2160, 3840)
                    frames.append(yuv)

                elif flags.img_type == 2:
                    offset += v
                    roi = cut_roi(frame, roi_upleft, roi_wh, offset)
                    frame = assign_roi(roi, roi_upleft)
                    yuv = BGR2yuv420sp(frame, 1080, 1920)
                    gray = yuv[:1080]
                    cv2.imwrite(f"img_{frame_id:04d}.jpg", gray)
                    frames.append(gray)


            if  frame_id >= end_frame:
                break

        cap.release()

        with open(flags.out_file, "wb") as f:
            for frame in frames:
                frame.tofile(f)






