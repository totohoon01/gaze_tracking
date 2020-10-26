import dlib
import cv2
import numpy as np

face_dectector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def set_trackbar(winname, valname, minval=0, maxval=255):
    def onChange(x):
        pass
    cv2.namedWindow(winname)
    cv2.createTrackbar(valname, winname, minval, maxval, onChange)

def get_face(frame):
    temp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_dectector(temp_frame)
    if faces:
        face = faces[0]
        #cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0,0,255), 3)
        return face

def get_face_landmarks(frame, face):
    landmarks = landmark_predictor(frame, face)
    #for p in landmarks.parts():
    #    cv2.circle(frame, (p.x, p.y), 1, (255,0,0), 3)
    return landmarks.parts()

def get_eye_region(frame, landmarks, LR="L"):
    LEFT_EYE = [36,37,39,40,41]
    RIGHT_EYE = [42,42,44,45,46,47]
    points = LEFT_EYE if LR == "L" else RIGHT_EYE
    region = np.array([(landmarks[point].x, landmarks[point].y) for point in points])
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [region], (255,255,255))
    eye_region = cv2.bitwise_and(frame, mask)
    cv2.imshow("eye", eye_region)
    return eye_region

set_trackbar("Thresh", "thr")
def get_center_of_pupil_from_eye(eye_region):
    eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    eye_gray = cv2.bilateralFilter(eye_gray, 10, 15, 15)
    eye_gray = cv2.erode(eye_gray, kernel, iterations=3)
    thr = cv2.getTrackbarPos("thr", "Thresh")
    eye_gray = cv2.threshold(eye_gray, thr, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("Thresh", eye_gray)
    return eye_gray