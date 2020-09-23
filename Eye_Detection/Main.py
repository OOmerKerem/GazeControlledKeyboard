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

COUNTER = 0
TOTAL = 0

font = cv2.FONT_HERSHEY_PLAIN

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    ret, frame = cap.read()

    faces = detector(frame)
    for face in faces:
        face_x, face_y = face.left(), face.top()
        face_x1, face_y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (face_x, face_y), (face_x1, face_y1), (0, 0, 255), 2)

        landmarks = predictor(frame, face)

        r_blink_ratio = blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        l_blink_ratio = blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        avg_blink_ratio = (l_blink_ratio + r_blink_ratio) / 2.0
        #print(avg_blink_ratio)


        right_eye_region = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                    (landmarks.part(37).x, landmarks.part(37).y),
                                    (landmarks.part(38).x, landmarks.part(38).y),
                                    (landmarks.part(39).x, landmarks.part(39).y),
                                    (landmarks.part(40).x, landmarks.part(40).y),
                                    (landmarks.part(41).x, landmarks.part(41).y)], np.int32)

        min_x = np.min(right_eye_region[:, 0])
        max_x = np.max(right_eye_region[:, 0])
        min_y = np.min(right_eye_region[:, 1])
        max_y = np.max(right_eye_region[:, 1])

        right_eye = frame[min_y:max_y , min_x:max_x]
        right_eye = cv2.resize(right_eye , None, fx=5, fy=5)
        gray_right_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
        #equ_right_eye =cv2.equalizeHist(gray_right_eye)
        #print(equ_right_eye)
        _, thresholded_right_eye = cv2.threshold(gray_right_eye, 37, 255, cv2.THRESH_BINARY)
        cv2.imshow("Gray Right Eye", gray_right_eye)
        cv2.imshow("Right Eye", thresholded_right_eye)

        if (avg_blink_ratio<0.2):
            COUNTER += 1

        else :
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
