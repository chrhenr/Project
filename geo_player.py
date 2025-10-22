import pyautogui
import time
from PIL import ImageGrab

import math
import geo_network
import torch


from map_plot import MapPlotter
from main import GeoGuesserHelper
import s2sphere


class GeoGuesserPlayer:
    def __init__(self):
        self.top_left_corner = (0, 89)
        self.bottom_right_corner = (1918, 1016)
        self.small_map = (1660, 841)
        self.middle_map = (1287, 568)

        self.mapcoords = (726, 209, 1749, 962)  # x1, y1, x2, y2

        self.coords = (-180.0, 180.0, -85.80112878, 85.80112878)  # lon_min, lon_max, lat_min, lat_max

    

    def screenshots(self):
        img = ImageGrab.grab(bbox=(self.top_left_corner[0], self.top_left_corner[1], self.bottom_right_corner[0], self.bottom_right_corner[1]))
        img.save("screenshot.png")
        return img

    def move_to_map(self):
        pyautogui.moveTo(self.small_map, duration=0.5)

    def center_map(self, x, y):
        pyautogui.moveTo(x, y, duration=1)
        time.sleep(0.5)
        pyautogui.click()

    def close_screenshot_tool(self):
        pyautogui.moveTo(1878, 884, duration=0.5)
        pyautogui.click()


    def geo_guesser_map(self):
        x1, y1, x2, y2 = self.mapcoords

        x_len = x2 - x1
        y_len = y2 - y1


        x_mid = x_len // 2 + x1
        y_mid = y_len // 2 + y1

        return x_len, y_len, x_mid, y_mid



    def from_latlon_to_pixel(self, lat, lon, x_len, y_len):
        # Define your map's approximate visible geographic bounds
        lon_min, lon_max, lat_min, lat_max = self.coords

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



    def predict_image(self, img, helper: GeoGuesserHelper):

        ## Load the trained model and make prediction
        model = geo_network.model_ResNet50()
        model.load_state_dict(torch.load(helper.save_path, map_location=torch.device('cpu')))
        model.eval()

        preds = helper.test_single_image(model, img)

        #Convert to lat lon
        index = torch.argmax(preds, dim=1).item()
        cell = helper.idx_to_cell[index]

        s2sphere_cell = s2sphere.CellId(cell)
        latLon = s2sphere.LatLng.from_point(s2sphere.Cell(s2sphere_cell).get_center())

        lat = latLon.lat().degrees
        lon = latLon.lng().degrees

        return lat, lon, preds


def player():
    player = GeoGuesserPlayer()
    helper = GeoGuesserHelper(recompute=False, vm=False)
    plotter = MapPlotter(helper.df)
    helper.prepare_data()

    ##Start the geo guesser map interaction
    img = player.screenshots()
    time.sleep(1)
    player.close_screenshot_tool()
    time.sleep(1)
    player.move_to_map()
    time.sleep(1)
    x_len, y_len, x_mid, y_mid = player.geo_guesser_map()
    player.center_map(x_mid, y_mid)
    time.sleep(1)

    ## Make prediction
    lat, lon, preds = player.predict_image(img, helper)
    x_pixel, y_pixel = player.from_latlon_to_pixel(lat, lon, x_len, y_len)

    lat, lon = -33.918861, 18.423300  # Cape Town
    x_pixel, y_pixel = player.from_latlon_to_pixel(lat, lon, x_len, y_len)
    

    # Move mouse to that position (centered around map midpoint)
    pyautogui.moveTo(x_pixel + x_mid - x_len // 2, y_pixel + y_mid - y_len // 2, duration=1)
    pyautogui.click()

    # Plot the prediction on the map
    print(torch.argmax(preds, dim=1).item())
    plotter.plot_predictions(preds)




if __name__ == "__main__":
    player()

    