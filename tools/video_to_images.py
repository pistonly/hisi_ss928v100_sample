import cv2
from pathlib import Path
import argparse


argparser = argparse.ArgumentParser()
argparser.add_argument("--file", type=str, default="test.mp4", help="input video")
argparser.add_argument("--from_frame", type=int, default=0, help="from which frame to save")
argparser.add_argument("--frame_num", type=int, default=100, help="total frame number")
argparser.add_argument("--img_h", type=int, default=1080, help="resize frame height to img_h")
argparser.add_argument("--img_w", type=int, default=1920, help="resize frame width to img_w")
argparser.add_argument("--out_file", type=str, default="out.bin", help="output file")
argparser.add_argument("--img_type", type=int, default=0, help="0: gray, 1: yuv420sp")

flags = argparser.parse_args()

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
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    h, w = gray_frame.shape[:2]
                    if h != flags.img_h or w != flags.img_w:
                        gray_frame = cv2.resize(gray_frame, (flags.img_w, flags.img_h))
                    frames.append(gray_frame)
                    cv2.imwrite(f"img_{frame_id:04d}.jpg", gray_frame)
                elif flags.img_type == 1:
                    yuv = BGR2yuv420sp(frame, flags.img_h, flags.img_w)
                    frames.append(yuv)

            if  frame_id >= end_frame:
                break

        cap.release()

        with open(flags.out_file, "wb") as f:
            for frame in frames:
                frame.tofile(f)






