import cv2
import mediapipe as mp
import numpy as np
import time
from screenCalculations import *


# Camera position
bottom = True
# Grid parameters
num_cells = 3
grid_size = (num_cells, num_cells)
grid = np.zeros(shape=grid_size)


# Main code
mp_face_mesh = mp.solutions.face_mesh
mp_face_det = mp.solutions.face_detection

face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detection = mp_face_det.FaceDetection(min_detection_confidence=0.7)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()

    start = time.time()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(image)
    det_result = face_detection.process(image)
   
    obj_width = 0
    if det_result.detections:
        for detection in det_result.detections:

            bBox = detection.location_data.relative_bounding_box
            h, w, c = image.shape
            x, y, w, h = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)
            obj_width = w

    # To improve performance
    image.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])       
            
            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w

            # The camera matrix
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                  [0, focal_length, img_w / 2],
                                  [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            face_width = 14.0
            distance = distance_finder(focal_length, face_width, obj_width)
            x_limit, y_limit, cell_x, cell_y = limit_angles(distance, grid_size)

            print(y_limit)
            # Calculation of indices -> State grid
            idx = int(abs(x) / cell_x)
            idy = int((y + y_limit) / cell_y)

            # See where the user's head tilting

            if (y < -1 * y_limit) or (y >= (y_limit - y_limit / 3)) or (x < 0) or (x >= 2 * x_limit):
                text = "Outside"
            else:
                text = "Inside (%d, %d)" % (idx % grid_size[0], idy % grid_size[0])
                grid[idx % grid_size[0]][idy % grid_size[0]] = 1
            '''
            if y < -y_limit:
                text = "Outside Left"
            elif y > y_limit:
                text = "Outside Right"
            elif x < 0:
                text = "Outside Down"
            elif x > x_limit:
                text = "Outside Up"
            else:
                text = "Inside (%d, %d)" % (idx % grid_size[0], idy % grid_size[0])
                grid[idx % grid_size[0]][idy % grid_size[0]] = 1
            '''
            print(grid)
            grid[:][:] = 0
            
            # Clearing the Screen
            # os.system('clear')
            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
            
            cv2.line(image, p1, p2, (255, 0, 0), 3)

            # Add the text on the image
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
        # print("Focused: ", fps)

        cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    # connections=mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

    cv2.imshow('Head Pose Estimation', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break


cap.release()
