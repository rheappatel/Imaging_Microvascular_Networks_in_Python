import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from geomdl import fitting
from matplotlib.widgets import Button
import math
from math import pi
from numpy.linalg import norm


global circle_points
circle_points = np.array([])

with open("tubePoints.txt", 'w') as f:
    pass

with open("coordinates.txt", 'w') as f:
    pass


# Load the image
img = np.array(Image.open('microvascular.png'))



# Create a figure and axis with the image as background
fig, ax = plt.subplots()
ax.imshow(img)



# Lists to store all coordinates and current points
all_coords = []
points = []



# Plotting initial empty data
scat, = ax.plot([], [], 'o', label='Data Points')



user_width = int(input("What is the width? (micrometers) " ))
user_height = int(input("What is the height? (micrometers) "))



filepath = "microvascular.png"
img = Image.open(filepath)
 
# get width and height
img_width = img.width
img_height = img.height

width_ratio = user_width/img_width
height_ratio = user_height/img_height

def scale_x(x_coord):
    final_x = width_ratio * x_coord
    return final_x

def scale_y(y_coord):
    final_y = height_ratio * y_coord
    return final_y



def numerical_derivative(x, y, points):
    # Compute numerical derivative at point (x, y)
    if len(points) < 2:
        return None  # Need at least two points to compute derivative

    # Find the closest point in the list to (x, y)
    closest_point_idx = np.argmin(np.sqrt((points[:, 0] - x)**2 + (points[:, 1] - y)**2))
   
    # Use central difference method to calculate derivative
    if closest_point_idx == 0:
        # Forward difference
        dx = points[1, 0] - points[0, 0]
        dy = points[1, 1] - points[0, 1]
    elif closest_point_idx == len(points) - 1:
        # Backward difference
        dx = points[-1, 0] - points[-2, 0]
        dy = points[-1, 1] - points[-2, 1]
    else:
        # Central difference
        dx = (points[closest_point_idx + 1, 0] - points[closest_point_idx - 1, 0]) / 2.0
        dy = (points[closest_point_idx + 1, 1] - points[closest_point_idx - 1, 1]) / 2.0
   
    if dx != 0:
        slope = dy / dx
    else:
        slope = float('inf')  # Vertical line
   
    return slope


def calculate_derivatives(points):
    derivatives = []
    for i in range(len(points)):
        x, y = points[i]
        derivative = numerical_derivative(x, y, np.array(points))
        derivative = derivative * -1
        derivatives.append(derivative)
    return derivatives



def on_click(event):
    # Check if the click is within the image bounds and not on the button
    if event.inaxes == ax and event.button == 1:
        x = event.xdata
        y = event.ydata
        if x is not None and y is not None:
            points.append((x, y))
           
            # Update scatter plot with new points
            scat.set_data(*zip(*points))
       
            # Redraw the plot
            plt.draw()
           
            # Write coordinates to file
            scaled_x = scale_x(x)
            scaled_y = scale_y(y)
            with open("coordinates.txt", "a") as file:
                file.write(f"{scaled_x},{scaled_y},{0} \n")



def create_spline(event):
    global points  # Declare points as global
    if len(points) > 1:
        # Perform curve fitting
        degree = 3  # cubic curve
        try:
            curve = fitting.interpolate_curve(points, degree)
            evalpts = np.array(curve.evalpts)
                   
            # Plot the fitted curve
            curve_line, = ax.plot(evalpts[:, 0], evalpts[:, 1], color='blue', label='Fitted Curve')
                   
            # Update scatter plot
            ax.scatter(*zip(*points), color='red', label='Data Points')


            # Calculate derivatives at each point after spline fitting
            derivatives = calculate_derivatives(evalpts)
           
            for i in range(len(evalpts)):
                x, y = evalpts[i]
                x = scale_x(x)
                y = scale_y(y)
               
                print(f"Derivative at ({x}, {y}): {derivatives[i]}")
                slope = derivatives[i]
                r = 10
                center = (x, y, 0)      

                circCoords = np.transpose(angle(slope, r, center))
               
              
                with open("tubePoints.txt", 'a') as f:
                    np.savetxt(f, circCoords)
           
               
        
        except ValueError as e:
            print(f"Error during curve fitting: {e}")



    # Store current points into all_coords
    if points:
        all_coords.append(points[:])  # Make a copy of points
        points.clear()
   
    # Clear scatter plot only
    scat.set_data([], [])
   
    # Redraw the plot
    plt.draw()



# Connect the event handler to the figure
fig.canvas.mpl_connect('button_press_event', on_click)



plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)



# Define button and its functionality
axes = plt.axes([0.81, 0.000001, 0.15, 0.075])
bnext = Button(axes, 'Create spline', color="gray")



def button_callback(event):
    # Function to handle button click
    create_spline(event)



def angle(slope, radius, center):
    n_points = 100
    alpha = np.linspace(0, 2 * math.pi, n_points)
   
    n = np.array([0, 0, 1])
    nx = np.array([1, slope, 0])


    x_c = center[0]
    y_c = center[1]
    z_c = center[2]


    x = np.zeros(n_points)
    y = np.zeros(n_points)
    z = np.zeros(n_points)


    x = radius*np.cos(alpha)
    y = radius*np.sin(alpha)


    coords = np.array([x,
                       y,
                       z])

    
    n_mag = norm(n)
    nx_mag = norm(nx)


    n = n * (1/n_mag)
    nx = nx * (1/n_mag)
   

    v = np.cross(n, nx)
    v_mag = norm(v)
    v = v / v_mag


    theta = math.acos((np.dot(nx,n)) / (n_mag * nx_mag)) # angle of rotation


    x = v[0]
    y = v[1]
    z = v[2]


    num1 = (x * math.sin(theta)) - (y * z * (1 - math.cos(theta)))
    den1 = 1 - ((x**2 + z**2)*(1 - math.cos(theta)))


    phi = math.atan2(num1, den1)

    sigma = math.asin((x * y * (1 - math.cos(theta))) + (z * math.sin(theta)))


    num2 = (y * math.sin(theta)) - (x * z * (1 - math.cos(theta)))
    den2 = 1 - ((y**2 + z**2) * (1 - math.cos(theta)))


    rho = math.atan2(num2, den2)


    
    Rx = np.array([[1, 0, 0],
        [0, math.cos(phi), math.sin(phi)],
        [0, (math.sin(phi) * -1), math.cos(phi)]])

    Ry = np.array([[math.cos(sigma), 0, (math.sin(sigma) * -1)],
        [0, 1, 0],
        [math.sin(sigma), 0, math.cos(sigma)]])

    Rz = np.array([[math.cos(rho), math.sin(rho), 0],
        [(math.sin(rho) * -1), math.cos(rho), 0],
        [0, 0, 1]])


    R = np.matmul(np.matmul(Rz, Ry), Rx)



    newCoords = np.matmul(R, coords)
   
    newCoords[0,:] = newCoords[0,:] + x_c
    newCoords[1,:] = newCoords[1,:] + y_c
    newCoords[2,:] = newCoords[2,:] + z_c

    return newCoords



bnext.on_clicked(button_callback)



plt.show()
