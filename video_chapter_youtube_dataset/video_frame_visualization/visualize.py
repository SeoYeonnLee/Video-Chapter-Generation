"""
Visualize the shot detection results. Expand video frames to a series of images

"""

import os
import cv2
import numpy as np
import urllib.request
from PIL import Image, ImageDraw, ImageFont


def visualization_frame_with_timestamps(video_path, timestamps, is_sparse=True):
    """
    It is used for visualizing timestamp video chapters

    """

    # video should be already existed
    if not os.path.exists(video_path):
        assert False, "Cannot find video %s!"%video_path

    # load video
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    count = 0
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    resized_width = width // 5
    resized_height = height // 5
    frames = []
    success = True
    while success:
        success, frame = vidcap.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (resized_width, resized_height))
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        if is_sparse:
            if count % round(fps) == 0:
                frames.append(resized_frame)
        else:
            frames.append(resized_frame)

        count += 1

    frames = np.stack(frames, axis=0)

    # get the cut/transition intervals
    threshold = 1.0
    predictions = np.zeros((len(frames)), np.float32)

    cut_start_indices = []
    cut_end_indices = []
    for timestamp in timestamps:
        if is_sparse:
            start_index = round(timestamp - 3)
            end_index = round(timestamp + 3)
        else:
            start_index = round((timestamp - 3) * fps)
            end_index = round((timestamp + 3) * fps)

        cut_start_indices.append(start_index)
        cut_end_indices.append(end_index)

        predictions[start_index] = threshold
        predictions[end_index] = threshold

    # display
    # you may modify some visualization parameters
    row_image_num = 65
    ih, iw, ic = frames.shape[1:]
    if len(frames) % row_image_num != 0:
        pad_with = row_image_num - len(frames) % row_image_num
        frames = np.concatenate([frames, np.zeros([pad_with, ih, iw, ic], np.uint8)])
        predictions = np.concatenate([predictions, np.zeros([pad_with], np.float32)])
    col_num = len(frames) // row_image_num

    scene = frames.reshape([col_num, row_image_num, ih, iw, ic])
    scene = np.concatenate(np.split(np.concatenate(np.split(scene, col_num), axis=2)[0], row_image_num), axis=2)[0]

    img = Image.fromarray(scene)
    draw = ImageDraw.Draw(img)

    start = True
    i = 0
    for h in range(col_num):
        for w in range(row_image_num):
            draw.line((w * iw + iw - 3, h * ih, w * iw + iw - 3, (h + 1) * ih), fill=(0, 0, 0), width=4)
            draw.line((w * iw, h * ih, (w + 1) * iw, h * ih), fill=(255, 255, 255))
            if predictions[i] >= threshold:
                if start:
                    draw.line((w * iw + iw - 3, h * ih + ih / 2 * (1 - predictions[i]), w * iw + iw - 3, h * ih + ih / 2 * (1 + predictions[i])), fill=(255, 0, 0), width=6)
                else:
                    draw.line((w * iw + iw - 3, h * ih + ih / 2 * (1 - predictions[i]), w * iw + iw - 3, h * ih + ih / 2 * (1 + predictions[i])), fill=(0, 255, 0), width=6)
                start = not start
            else:
                draw.line((w * iw + iw - 3, h * ih + ih / 2 * (1 - predictions[i]), w * iw + iw - 3, h * ih + ih / 2 * (1 + predictions[i])), fill=(0, 0, 0), width=6)
            i += 1

    return img




if __name__ == "__main__":
    video_path = "D:/youtube_video_dataset/-SUlsj5a6iw.mp4"
    timestamps = [102, 184, 249, 340, 442, 8*60+31, 605, 12*60+2, 13*60+4, 14*60+26, 15*60+43, 17*60+24, 18*60+54]
    # video_path = "D:/youtube_video_dataset/-L8yV0qabOY.mp4"
    # timestamps = [64, 2*60+45, 4*60+5, 5*60+16, 7*60+5]

    visual_img = visualization_frame_with_timestamps(video_path, timestamps, is_sparse=False)
    visual_img.save("D:/py3_code/video_chapter_youtube_dataset/dense_ex2.jpg")


