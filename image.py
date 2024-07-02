import cv2
import numpy as np
import torch
import base64


def load_image(uri):
    # img = cv2.imread("images/img124.png")
    encoded_data = uri.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, channels = img.shape
    return img, width, height


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def find_contours(image, thickness=3):
    contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_image = np.zeros_like(image)
    cv2.drawContours(contour_image, contours, -1, 255, thickness)
    return contour_image, contours, hierarchy


def extract_idcard(raw_image, mask_image):
    img = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    contour_image, contours, hierarchy = find_contours(img)

    cnts = sorted(contours, key=cv2.contourArea, reverse=True)
    screenCntList = []
    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        screenCnt = approx

        if (len(screenCnt) == 4):
            screenCntList.append(screenCnt)

    assert len(screenCntList) == 1
    new_points = np.array([[points[0][0], points[0][1]] for points in screenCntList[0]])

    warped = four_point_transform(raw_image, new_points)
    return cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), new_points

def unpad(image: np.array, pads) -> np.ndarray:
    x_min_pad, y_min_pad, x_max_pad, y_max_pad = pads
    height, width = image.shape[:2]

    return image[y_min_pad : height - y_max_pad, x_min_pad : width - x_max_pad]


def pad(image: np.array, factor: int = 32, border: int = cv2.BORDER_REFLECT_101) -> tuple:
    height, width = image.shape[:2]

    if height % factor == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = factor - height % factor
        y_min_pad = y_pad // 2
        y_max_pad = y_pad - y_min_pad

    if width % factor == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = factor - width % factor
        x_min_pad = x_pad // 2
        x_max_pad = x_pad - x_min_pad

    padded_image = cv2.copyMakeBorder(image, y_min_pad, y_max_pad, x_min_pad, x_max_pad, border)

    return padded_image, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)

def tensor_from_rgb_image(image: np.ndarray) -> torch.Tensor:
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    return torch.from_numpy(image)