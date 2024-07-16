import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.widgets import Button
from scipy.interpolate import make_interp_spline




# Load the microvascular network image
img = np.array(Image.open('microvascular.png'))


# Create a figure and axis with the image as background
fig, ax = plt.subplots()
ax.imshow(img)


# Lists to store x and y coordinates
x_coords = []
y_coords = []


# Plotting initial empty data
scat, = ax.plot([], [], 'o', label='Data Points')
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)




def on_click(event):
    # Append the coordinates of the mouse click
    x = float(event.xdata)
    y = float(event.ydata)
    x_coords.append(x)
    y_coords.append(y)
    print(x_coords)
   
    # Update the scatter plot with new points
    scat.set_data(x_coords, y_coords)
           
    # Redraw the plot
    plt.draw()
    for i in range(0, len(x_coords), 1):
        plt.plot(x_coords[i:i+2], y_coords[i:i+2], 'ro-')
    plt.axis('equal')
    plt.show()
   

# Connect the event handler to the figure
fig.canvas.mpl_connect('button_press_event', on_click)


# Show the plot
plt.show()
