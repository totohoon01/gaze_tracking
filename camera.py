import cv2
import landmarks as mk
cap = cv2.VideoCapture(0)

#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944); 
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)

while 1:
    ret, frame = cap.read()
    if ret:
        face = mk.get_face(frame)
        if face:
            landmarks = mk.get_face_landmarks(frame, face)
            
            left_in_corner = landmarks[39]
            left_out_corner = landmarks[36]
            right_in_corner = landmarks[42]
            right_out_corner = landmarks[45]
            eye = mk.get_eye_region(frame, landmarks)
            eye_gray = mk.get_center_of_pupil_from_eye(eye)
            
        cv2.imshow("Frame", frame)
        if cv2.waitKey(10) >= 0:
            break
cap.release()
cv2.destroyAllWindows()
 