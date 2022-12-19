import tkinter as tk
import math

def screenParameters():
    root = tk.Tk()

    W = root.winfo_screenmmwidth()/10
    H = root.winfo_screenmmheight()/10

    return W, H
    
def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    try:    
        distance = (real_face_width * Focal_Length)/face_width_in_frame
    except:
        print('Face not detected')
        distance = 51
            
    return distance 

def limitAngles(distance, grid_size = 1):
    
    W, H = screenParameters()

    W, H = screenParameters()
    #Assuming we have the camera on top of the screen ->
    dCam = distance

    theta_y = math.tan(W / (2 * dCam))
    theta_y = math.atan(theta_y) #-> Rad
    theta_y *= 180/math.pi

    theta_x = math.tan(H / (2 * dCam))
    theta_x = math.atan(theta_x) #-> Rad
    theta_x *= 180/math.pi

    cell_x = cell_y = 0

    if grid_size != 1:
        #Screen parameters & cell sizes for both axis        
        cell_size_x = H/ grid_size[0]
        cell_size_y = W /grid_size[0]

        cell_x = math.tan(cell_size_x / dCam)
        cell_x = math.atan(cell_x) #-> Rad
        cell_x *= 180 / math.pi

        cell_y = math.tan(cell_size_y / dCam)
        cell_y = math.atan(cell_y) #-> Rad
        cell_y *= 180 / math.pi


    return theta_x, theta_y, cell_x, cell_y
