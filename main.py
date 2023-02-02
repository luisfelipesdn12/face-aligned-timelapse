import glob
import os

import cv2
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

from utils import (c_closest, lm2coord, photo_date_formatted, photo_datetime,
                   photo_day_number, point_dist, rotate, shrink, to_target)

render_video_dir = "."
input_dir = "./input"
labled_dir = "./labled"
fps = 15

font = ImageFont.truetype("./fonts/Roboto.ttf", 30)

# Create dir if not exist
if not os.path.exists(labled_dir):
    os.makedirs(labled_dir)

FIRST_PHOTO_DATE = None
is_first_photo = True

# Apply label to images and save
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        with Image.open(os.path.join(input_dir, filename)) as image:
            draw = ImageDraw.Draw(image)

            if (is_first_photo):
                FIRST_PHOTO_DATE = photo_datetime(filename)

            day = photo_day_number(filename, FIRST_PHOTO_DATE)
            date = photo_date_formatted(filename)

            draw.text((40, 40), f"Dia {day} - {date}", (0, 0, 0), font=font)

            image.save(f"{labled_dir}/{day}.jpg")

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
# Adjust maximum number of faces if you took group photos
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=5, static_image_mode=True)

# Sort image names by number
def sort_key(str):
    num = str.replace(labled_dir + "/", "").replace(".jpg", "")
    return int(num)

img_names = glob.glob(labled_dir + "/*.jpg")
img_names.sort(key=sort_key)
first_img = cv2.imread(img_names[0])
(h, w, c) = first_img.shape
resolution = (w, h)
center = (int(resolution[0]/2), int(resolution[1]/2))

fnf = 0

############### FIND SMALLEST CENTER FACE ##############
# This is needed to resize every photo according to the one in which your face is the most distant to the camera.
# Why? So that no photo will be cropped out of the screen!

sm_eyedist = float("inf")

# This is the position every photo will be aligned to, the left eye of the smallest face found.
leyepos = "NO VALUE"

for filename in img_names:

    print("Checking smallest face in image", filename)

    img = cv2.imread(filename)
    img = cv2.resize(img, resolution, interpolation=cv2.INTER_AREA)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = faceMesh.process(imgRGB)

    faces = result.multi_face_landmarks

    if(faces):
        print("Detected", len(faces), "faces")

    face = c_closest(faces, center, resolution)

    if (face):
        nose = lm2coord(face.landmark[4], resolution)
        leye = lm2coord(face.landmark[133], resolution)
        reye = lm2coord(face.landmark[362], resolution)

        dist = point_dist(leye, reye)

        if (dist < sm_eyedist):
            sm_filename = filename
            sm_eyedist = dist
            leyepos = leye
    else:
        print("No faces found!")

print("#### Smallest face is in file", sm_filename)
print("#### Alignment position is at", leyepos)

############### BUILD VIDEO ##############

video = cv2.VideoWriter(render_video_dir + "/timelapse.mp4",
                        0x7634706d, fps, resolution)

for filename in img_names:
    print("Processing img " + filename)

    img = cv2.imread(filename)
    img = cv2.resize(img, resolution)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = faceMesh.process(imgRGB)

    face = c_closest(result.multi_face_landmarks, center, resolution)

    if (face):

        nose = lm2coord(face.landmark[4], resolution)
        leye = lm2coord(face.landmark[133], resolution)
        reye = lm2coord(face.landmark[362], resolution)

        # shrink
        eyedist = point_dist(leye, reye)
        img = shrink(img, leye, sm_eyedist/eyedist, resolution)

        # to target
        img = to_target(img, leye, leyepos, resolution)

        # rotate
        img = rotate(img, leye, leye, reye, resolution)

        video.write(img)
    else:
        print("Skipped image", filename, "because no faces were found")
        fnf += 1

print("Number of images with no faces found:", fnf)
video.release()
print("Video was rendered in", render_video_dir)
