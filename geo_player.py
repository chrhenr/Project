import pyautogui
import time
from PIL import ImageGrab

import math
import geo_network
import torch


from map_plot import MapPlotter
from main import GeoGuesserHelper
import s2sphere

from pytesseract import pytesseract as pyt


class GeoGuesserPlayer:
    def __init__(self):
        self.top_left_corner = (0, 89)
        self.bottom_right_corner = (1918, 1016)
        self.small_map = (1660, 841)
        self.middle_map = (1287, 568)
        self.next_button = (953, 954,)
        self.guess_button = (1257, 992)


        self.mapcoords = (641, 208, 1886, 962)  # x1, y1, x2, y2

        self.coords = (-180.0, 180.0, -85.0, 85.0)  # lon_min, lon_max, lat_min, lat_max

        self.kodiak_lon = -152.41789208250827
        self.kodiak_lat = 57.79491777998742
        self.kodiak_x = 802
        self.kodiak_y = 419

        self.hobart_lon = 147.3272
        self.hobart_lat = -42.8821
        self.hobart_x = 1656
        self.hobart_y = 753



    def screenshots(self):
        img = ImageGrab.grab(bbox=(self.top_left_corner[0], self.top_left_corner[1], self.bottom_right_corner[0], self.bottom_right_corner[1]))
        img.save("screenshot.png")
        return img

    def move_to_map(self):
        pyautogui.moveTo(self.small_map, duration=0.5)

    def center_map(self, x, y):
        pyautogui.moveTo(x, y, duration=1)
        time.sleep(0.5)

    def close_screenshot_tool(self):
        pyautogui.moveTo(1878, 884, duration=0.5)
        pyautogui.click()

    def next_round(self):
        pyautogui.moveTo(self.next_button, duration=0.5)
        pyautogui.click()
        time.sleep(2)

    def guess(self, x_pixel, y_pixel):
        pyautogui.moveTo(x_pixel, y_pixel, duration=1)
        pyautogui.click()
        time.sleep(1)
        pyautogui.moveTo(self.guess_button, duration=0.5)
        pyautogui.click()
        time.sleep(2)


    def geo_guesser_map(self):
        x1, y1, x2, y2 = self.mapcoords

        x_len = x2 - x1
        y_len = y2 - y1


        x_mid = x_len // 2 + x1
        y_mid = y_len // 2 + y1

        return x_len, y_len, x_mid, y_mid


    def lat_to_mercator_y(self, lat: float) -> float:
        return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    
    def from_latlon_to_pixel(self, lat, lon):
        # --- Longitude to X (linear scale) ---
        # Fraction of how far the target lon is between the refs
        frac_x = (lon - self.kodiak_lon) / (self.hobart_lon - self.kodiak_lon)
        x = frac_x * (self.hobart_x - self.kodiak_x) + self.kodiak_x

        # --- Latitude to Y (mercator scale) ---
        merc_kodiak = self.lat_to_mercator_y(self.kodiak_lat)
        merc_hobart = self.lat_to_mercator_y(self.hobart_lat)
        merc_target = self.lat_to_mercator_y(lat)

        frac_y = (merc_target - merc_kodiak) / (merc_hobart - merc_kodiak)
        y = frac_y * (self.hobart_y - self.kodiak_y) + self.kodiak_y


        return round(x), round(y)
        


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


    def test_area():
        player = GeoGuesserPlayer()

        pyautogui.moveTo(player.kodiak_x, player.kodiak_y, duration=1)
        time.sleep(1)
        pyautogui.click()
        time.sleep(1)
        pyautogui.moveTo(player.hobart_x, player.hobart_y, duration=1)
        time.sleep(1)
        pyautogui.click()

    def play_single_round(self, helper: GeoGuesserHelper, plotter: MapPlotter):
        ##Start the geo guesser map interaction
        img = self.screenshots()
        time.sleep(1)
        self.move_to_map()
        time.sleep(1)
        x_mid, y_mid, _, _ = self.geo_guesser_map()
        self.center_map(x_mid, y_mid)
        time.sleep(1)

        ## Make prediction
        lat, lon, preds = self.predict_image(img, helper)
        x_pixel, y_pixel = self.from_latlon_to_pixel(lat, lon)

        self.guess(x_pixel, y_pixel)

        self.next_round()

        # plotter.plot_predictions(preds)
        # plotter.plot_embellished_predictions(preds)
    

def player():
    player = GeoGuesserPlayer()
    helper = GeoGuesserHelper(recompute=False, vm=False)
    plotter = MapPlotter(helper.df)
    helper.prepare_data()

    for _ in range(5):
        player.play_single_round(helper, plotter)

    
    # Plot the prediction



#Geoguesser: 6407, 7558, 8664, 11954, 8674
#Vi: 7258, 1382, 9671, 10801, 14408
avg_vi = (7258 + 1382 + 9671 + 10801 + 14408) / 5
print("Our average score:", avg_vi)
avg_geo = (6407 + 7558 + 8664 + 11954 + 8674) / 5
print("GeoGuessr average score:", avg_geo)

if __name__ == "__main__":
    player()

    