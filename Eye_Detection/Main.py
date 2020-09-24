import cv2
import numpy as np
import dlib
from math import hypot


def midpoint(n1, n2):
    return int((n1.x + n2.x) / 2), int((n1.y + n2.y) / 2)


def blinking_ratio(eye_points, face_landmarks):
    eye_left = (face_landmarks.part(eye_points[0]).x, face_landmarks.part(eye_points[0]).y)
    eye_right = (face_landmarks.part(eye_points[3]).x, face_landmarks.part(eye_points[3]).y)
    eye_top = midpoint(face_landmarks.part(eye_points[1]), face_landmarks.part(eye_points[2]))
    eye_bottom = midpoint(face_landmarks.part(eye_points[5]), face_landmarks.part(eye_points[4]))

    cv2.rectangle(frame, (eye_left[0] - 10, eye_bottom[1] - 20), (eye_right[0] + 10, eye_top[1] + 20),
                  (0, 255, 255), 2)

    hor_lenght = hypot((eye_left[0] - eye_right[0]), (eye_left[1] - eye_right[1]))
    ver_lenght = hypot((eye_top[0] - eye_bottom[0]), (eye_bottom[1] - eye_top[1]))

    ratio = ver_lenght / hor_lenght

    return ratio


def gaze_direction(eye_points, face_landmarks):
    eye_region = np.array([(face_landmarks.part(eye_points[0]).x, face_landmarks.part(eye_points[0]).y),
                           (face_landmarks.part(eye_points[1]).x, face_landmarks.part(eye_points[1]).y),
                           (face_landmarks.part(eye_points[2]).x, face_landmarks.part(eye_points[2]).y),
                           (face_landmarks.part(eye_points[3]).x, face_landmarks.part(eye_points[3]).y),
                           (face_landmarks.part(eye_points[4]).x, face_landmarks.part(eye_points[4]).y),
                           (face_landmarks.part(eye_points[5]).x, face_landmarks.part(eye_points[5]).y)], np.int32)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [eye_region], True, 255, 2)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    gray_eye = eye[min_y:max_y, min_x:max_x]
    _, thresholded_eye = cv2.threshold(gray_eye, 53, 255, cv2.THRESH_BINARY)
    thresholded_eye = cv2.resize(thresholded_eye, None, fx=5, fy=5)
    cv2.imshow("Eye", thresholded_eye)

    eye_height, eye_width= thresholded_eye.shape
    left_side_eye = thresholded_eye[0:height, 0:int(width/2)]
    right_side_eye = thresholded_eye[0:height, int(width/2):width]

    size = np.size(thresholded_eye)
    white = np.count_nonzero(thresholded_eye)

    print("white : ", eye_height)
    print("size : ", eye_width)


COUNTER = 0
TOTAL = 0

font = cv2.FONT_HERSHEY_PLAIN

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
r_eye_region = [36, 37, 38, 39, 40, 41]
l_eye_region = [42, 43, 44, 45, 46, 47]

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(frame)
    for face in faces:
        face_x, face_y = face.left(), face.top()
        face_x1, face_y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (face_x, face_y), (face_x1, face_y1), (0, 0, 255), 2)

        landmarks = predictor(frame, face)

        r_blink_ratio = blinking_ratio(r_eye_region, landmarks)
        l_blink_ratio = blinking_ratio(l_eye_region, landmarks)
        avg_blink_ratio = (l_blink_ratio + r_blink_ratio) / 2.0

        gaze_direction(r_eye_region, landmarks)

        if (avg_blink_ratio < 0.2):
            COUNTER += 1

        else:
            if COUNTER >= 4:
                TOTAL += 1

            COUNTER = 0

        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30), font, 3, (0, 255, 0), 2)

    cv2.imshow("Ekran", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
