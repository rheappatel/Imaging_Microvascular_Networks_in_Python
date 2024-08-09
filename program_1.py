import cv2
import numpy as np

file = input("Paste file path: ")
img = cv2.imread(file)
rows, cols = img.shape[:2]
gaussian_blur = cv2.GaussianBlur(img, (7,7), 2)
sharpened3 = cv2.addWeighted(img,7.5,gaussian_blur, -6.5,0)

def get_Coord_Dimen_x(a, x, img_width):
    return (x / img_width) * a

def get_Coord_Dimen_y(b, y, img_height):
    return (y / img_height) * b


def find_centerlines(binary_image_path):
    binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)
    skeleton = cv2.ximgproc.thinning(binary_image, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(output_image, (cX, cY), 5, (0, 0, 255), -1)
    cv2.imwrite('segmented_vessels.png', output_image)
    cv2.imshow('Segmented Vessels and Centerlines', output_image)

def code():
    cv2.imwrite("sharpened3.png", sharpened3)
    supersharp = cv2.imread("/Users/daisymaturo/Downloads/microvascularpython/sharpened3.png")
    output_box = cv2.boxFilter(supersharp, -1, (5,5), normalize=False)
    cv2.imshow("original", img)
    cv2.imwrite("box_wo_blur.png", output_box)
    cv2.imshow("sharpened", sharpened3)
    cv2.imshow("box filtered & sharpened", output_box)
    image_color = cv2.imread("/Users/daisymaturo/Downloads/microvascularpython/box_wo_blur.png",cv2.IMREAD_GRAYSCALE)
    thres = 225
    img_bw = cv2.threshold(image_color, thres, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("box, sharp bw", img_bw)
    cv2.imwrite("box_sharp_bw.png", img_bw)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error loading image {image_path}")
        return
        
    _, binary_image = cv2.threshold(image, 225, 255, cv2.THRESH_BINARY)
    cv2.imwrite("binaryimg.png", binary_image)
    print(f"Binary image saved as binaryimg.png")
    find_centerlines("/Users/daisymaturo/Downloads/microvascularpython/binaryimg.png")

def record_white_pixel_coordinates(image_path, output_file, height_um):
    code()
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error loading image {image_path}")
        return
    _, binary_image = cv2.threshold(image, 225, 255, cv2.THRESH_BINARY)
    white_pixel_coords = np.column_stack(np.where(binary_image == 255))
    aspect_ratio = image.shape[1] / image.shape[0]
    width_um = height_um * aspect_ratio
    physical_coords = [(get_Coord_Dimen_x(width_um, x, image.shape[1]), get_Coord_Dimen_y
                        (height_um, y, image.shape[0])) for y, x in white_pixel_coords]
    with open(output_file, 'w') as f:
        for coord in physical_coords:
            f.write(f"{coord[0]},{coord[1]}\n")
    print(f"Physical coordinates of white pixels have been saved to '{output_file}'")

image_path = "/Users/daisymaturo/Downloads/microvascularpython/box_sharp_bw.png"
height_um = int(input("Height of the image in physical dimensions (um): "))
output_file = 'white_pixel_coords.txt'


record_white_pixel_coordinates(image_path, output_file, height_um)

cv2.waitKey(0)
