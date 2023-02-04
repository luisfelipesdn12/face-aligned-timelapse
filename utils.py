from datetime import datetime
import re
import cv2
from math import atan2, degrees, dist
import numpy as np


def photo_datetime(photo_date: str):
    """
    Args:
        photo_date: Telegram's datetime of the photo.

    Returns the datetime of the photo.
    """

    regex_date = re.compile(r"photo_\d+@(\d{2}-\d{2}-\d{4})_\d{2}-\d{2}-\d{2}.jpg")
    date_str = regex_date.search(photo_date).group(1)

    day, month, year = [int(i) for i in date_str.split("-")]

    return datetime(year, month, day)


def photo_day_number(photo_date: str, FIRST_PHOTO_DATE: datetime):
    """
    Args:
        photo_date: Telegram's datetime of the photo.

    Returns the number of days since the first
    photo.
    """

    date = photo_datetime(photo_date)

    return (date - FIRST_PHOTO_DATE).days + 1


def photo_date_formatted(photo_date: str):
    """
    Args:
        photo_date: Telegram's datetime of the photo.

    Returns the formatted string of the date.
    e.g.: `"17/03/2021"`
    """
    date = photo_datetime(photo_date)

    return f"{date.day:02}/{date.month:02}/{date.year:04}"


def lm2coord(landmark, resolution):
    """Converts a face landmark to coordinates"""

    width, height = resolution
    return (int(landmark.x * width), int(landmark.y * height))


def to_target(img, p, t, resolution):
    """Moves an image overlapping point p to point t"""

    xc, yc = t
    xp, yp = p
    offx = xc - xp
    offy = yc - yp

    M = np.float32([[1, 0, offx], [0, 1, offy]])

    return cv2.warpAffine(img, M, resolution)


def shrink(img, pivot, scale, resolution):
    """Shrinks an image based on the 'scale' value (between 0 and 1)"""

    M = cv2.getRotationMatrix2D(pivot, 0, scale)

    return cv2.warpAffine(img, M, resolution)


def rotate(img, pivot, ple, pre, resolution):
    """
    Calculates the degrees needed to rotate the image so that
    the eyes are in a straight line. Rotates around the pivot
    """

    deg = -degrees(atan2(ple[0] - pre[0], ple[1] - pre[1])) - 90

    M = cv2.getRotationMatrix2D(pivot, deg, 1)

    return cv2.warpAffine(img, M, resolution)


def at_center(p, resolution):
    """
    Checks if a point is at the center of the image
    (inside the second third of the height and width)
    """

    px, py = p
    width, height = resolution
    return (width / 3 <= px <= width * 2 / 3) and (height / 3 <= py <= height * 2 / 3)


def c_closest(faces, center, resolution):
    """
    Returns the face that is closest to the center
    among a given set
    """

    mindist = float("inf")
    minface = False

    if faces:
        for face in faces:
            nose = lm2coord(face.landmark[4], resolution)
            if at_center(nose, resolution):
                distance = dist(center, nose)
                if distance < mindist:
                    mindist = distance
                    minface = face

    return minface


def drawp(i, p):
    """(UTILITY) Draws a green dot in the specified position of the image"""

    out = cv2.circle(i, p, 5, (0, 255, 0), 5)

    cv2.imshow("drawp", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
