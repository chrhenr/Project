import pyautogui
import time
from PIL import ImageGrab

import math
import geo_network
import torch



top_left_corner = (0, 89)
bottom_right_corner = (1918, 1016)

small_map = (1660, 841)
middle_map = (1287, 568)


def screenshot():
    img = ImageGrab.grab(bbox=(top_left_corner[0], top_left_corner[1], bottom_right_corner[0], bottom_right_corner[1]))
    img.save("screenshot.png")
    return img

def move_to_map():
    pyautogui.moveTo(small_map, duration=0.5)

def center_map(x, y):
    pyautogui.moveTo(x, y, duration=1)
    time.sleep(0.5)
    pyautogui.click()

def close_screenshot_tool():
    pyautogui.moveTo(1878, 884, duration=0.5)
    pyautogui.click()


def geo_guesser_map():
    x1 = 726
    y1 = 209
    x2 = 1749
    y2 = 962

    x_len = x2 - x1
    y_len = y2 - y1

    print(f"x_len: {x_len}, y_len: {y_len}")

    x_mid = x_len // 2 + x1
    y_mid = y_len // 2 + y1

    print(f"x_mid: {x_mid}, y_mid: {y_mid}")
    return x_len, y_len, x_mid, y_mid



def from_latlon_to_pixel(lat, lon, x_len, y_len):
    # Define your map's approximate visible geographic bounds
    lon_min, lon_max = -180.0, 180.0
    lat_min, lat_max = -85.80112878, 85.80112878

    # Convert lat to radians
    lat_rad = math.radians(lat)
    lat_min_rad = math.radians(lat_min)
    lat_max_rad = math.radians(lat_max)

    # X coordinate (linear with longitude)
    x = (lon - lon_min) / (lon_max - lon_min) * x_len

    # Y coordinate (Mercator projection)
    def mercator_y(lat_r):
        return math.log(math.tan(lat_r / 2 + math.pi / 4))

    y = (mercator_y(lat_max_rad) - mercator_y(lat_rad)) / (mercator_y(lat_max_rad) - mercator_y(lat_min_rad)) * y_len

    # Return pixel offsets relative to top-left
    return int(x), int(y)



save_path = "./geo_network_test.pth"

def predict_image(img):
    
    model = geo_network.model_ResNet50()
    model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
    model.eval()

    




    return 37.7749, -122.4194  # San Francisco

def player():
    ##Start the geo guesser map interaction
    img = screenshot()
    time.sleep(1)
    close_screenshot_tool()
    time.sleep(1)
    move_to_map()
    time.sleep(1)
    x_len, y_len, x_mid, y_mid = geo_guesser_map()
    center_map(x_mid, y_mid)
    time.sleep(1)

    ## Make prediction
    lat, lon = predict_image(img)
    x_pixel, y_pixel = from_latlon_to_pixel(lat, lon, x_len, y_len)

    # Move mouse to that position (centered around map midpoint)
    pyautogui.moveTo(x_pixel + x_mid - x_len // 2, y_pixel + y_mid - y_len // 2, duration=1)
    pyautogui.click()



if __name__ == "__main__":
    player()

    