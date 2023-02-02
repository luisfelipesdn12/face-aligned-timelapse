from datetime import datetime
import re
import cv2
import math
import numpy as np

def photo_datetime(photo_date: str):
    """
    Args:
        - Telegram's datetime of the photo.

    Returns the datetime of the photo.
    """

    regex_date = re.compile(
        r"photo_\d+@(\d{2}-\d{2}-\d{4})_\d{2}-\d{2}-\d{2}.jpg")
    date_str = regex_date.search(photo_date).group(1)

    day, month, year = [int(i) for i in date_str.split("-")]

    return datetime(year, month, day)

def photo_day_number(photo_date: str, FIRST_PHOTO_DATE: datetime):
    """
    Args:
        - Telegram's datetime of the photo.

    Returns the number of days since the first
    photo.
    """

    date = photo_datetime(photo_date)

    return (date - FIRST_PHOTO_DATE).days + 1


def photo_date_formatted(photo_date: str):
    regex_date = re.compile(
        r"photo_\d+@(\d{2}-\d{2}-\d{4})_\d{2}-\d{2}-\d{2}.jpg")
    date_str = regex_date.search(photo_date).group(1)

    day, month, year = [int(i) for i in date_str.split("-")]

    return f"{day:02}/{month:02}/{year:04}"


def point_dist(a, b):
    """Returns the distance between two points"""

    xa, ya = a
    xb, yb = b
    diffx = abs(xa-xb)
    diffy = abs(ya-yb)
    return math.sqrt((diffx**2) + (diffy**2))


def lm2coord(lm, resolution):
    """Converts a face landmark to coordinates"""

    w, h = resolution
    return (int(lm.x*w), int(lm.y*h))


def to_target(img, p, t, resolution):
    """Moves an image overlapping point p to point t"""

    xc, yc = t
    xp, yp = p
    offx = abs(xp - xc) if (xp <= xc) else -abs(xp - xc)
    offy = abs(yp - yc) if (yp <= yc) else -abs(yp - yc)

    M = np.float32([[1, 0, offx], [0, 1, offy]])
    out = cv2.warpAffine(img, M, resolution)
    return out


def shrink(img, pivot, scale, resolution):
    """Shrinks an image based on the 'scale' value (between 0 and 1)"""

    M = cv2.getRotationMatrix2D(pivot, 0, scale)
    out = cv2.warpAffine(img, M, resolution)
    return out


def rotate(img, pivot, ple, pre, resolution):
    """Calculates the degrees needed to rotate the image so that the eyes are in a straight line. Rotates around the pivot"""

    deg = -90 - math.atan2(ple[0] - pre[0], ple[1] - pre[1]) * 180 / math.pi

    M = cv2.getRotationMatrix2D(pivot, deg, 1)
    out = cv2.warpAffine(img, M, resolution)
    return out


def at_center(p, resolution):
    """Checks if a point is at the center of the image (inside the second third of the height and width)"""

    px, py = p
    rw, rh = resolution
    return (rw/3 <= px <= rw*2/3) and (rh/3 <= py <= rh*2/3)


def c_closest(faces, center, resolution):
    """Returns the face that is closest to the center among a given set"""

    mindist = float("inf")
    minface = False

    if (faces):
        for face in faces:
            nose = lm2coord(face.landmark[4], resolution)
            if at_center(nose, resolution):
                dist = point_dist(center, nose)
                if (dist < mindist):
                    mindist = dist
                    minface = face

    return minface


def drawp(i, p):
    """(UTILITY) Draws a green dot in the specified position of the image"""

    out = cv2.circle(i, p, 5, (0, 255, 0), 5)

    cv2.imshow("drawp", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    example = "photo_1@20-03-2021_00-18-52"
    print(photo_day_number(example))  # 4
