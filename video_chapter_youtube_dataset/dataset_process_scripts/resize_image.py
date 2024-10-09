import cv2
import glob


img_dirs = glob.glob("E:/youtube_video_frame_dataset/*")
target_size = 96

for idx, img_dir in enumerate(img_dirs):
    print(f"process {idx}/{len(img_dirs)}..., {img_dir}")
    img_paths = glob.glob(img_dir + "/*.jpg")
    for img_path in img_paths:
        img = cv2.imread(img_path)
        resized_img = cv2.resize(img, (target_size, target_size))

        cv2.imwrite(img_path, resized_img)

