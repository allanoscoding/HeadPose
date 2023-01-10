import tkinter as tk
import math


def screen_parameters():
    root = tk.Tk()

    w = root.winfo_screenmmwidth()/10
    h = root.winfo_screenmmheight()/10

    return w, h


def distance_finder(focal_length, real_face_width, face_width_in_frame):

    try:    
        distance = (real_face_width * focal_length) / face_width_in_frame
    except:
        print('Face not detected')
        distance = 60
            
    return distance 


def limit_angles(distance, grid_size=(1, 1)):

    w, h = screen_parameters()
    # Assuming we have the camera on top of the screen ->
    d_cam = distance

    theta_y = math.tan(w / d_cam)
    # -> Rad
    theta_y = math.atan(theta_y)
    theta_y *= 180/math.pi

    theta_x = math.tan(h / (2 * d_cam))
    # -> Rad
    theta_x = math.atan(theta_x)
    theta_x *= 180/math.pi

    cell_x = cell_y = 0

    if grid_size[0] != 1:
        # Screen parameters & cell sizes for both axis
        cell_size_x = h / grid_size[0]
        cell_size_y = w / grid_size[0]

        cell_x = math.tan(cell_size_x / d_cam)
        # -> Rad
        cell_x = math.atan(cell_x)
        cell_x *= 180 / math.pi

        cell_y = math.tan(cell_size_y / d_cam)
        # -> Rad
        cell_y = math.atan(cell_y)
        cell_y *= 180 / math.pi

    return theta_x, theta_y, cell_x, cell_y
