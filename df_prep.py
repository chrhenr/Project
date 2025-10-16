from fileinput import filename
from itertools import takewhile
import os
import pandas as pd
from pathlib import Path
import json

# from torch.utils.data import Dataset, DataLoader
from PIL import Image
from s2sphere import CellId, LatLng, Cell
import matplotlib.pyplot as plt


CITIES_PATH = "./cities"
STREETVIEW_PATH = "./streetview"
STREETVIEW_JPG_PATH = "./streetview_jpg"
MAPPED_PATH = "./data_mapped"

def load_data(streetview_path, cities_path, mapped_path, recompute: bool) -> pd.DataFrame:
    if recompute == False:
        return pd.read_csv("data.csv")
    # Filvägar
    streetview_coords_path = streetview_path + "/coords.csv"

    cities_image_path = cities_path + "/Images"
    cities_dataframes_path = cities_path + "/Dataframes"
    cities_dataframes_dir = Path(cities_dataframes_path)
    csv_files = sorted(cities_dataframes_dir.glob("*.csv"))
    #läs in data fron data_mapped
    mapped_path = Path(mapped_path)
    json_files = list(mapped_path.glob("*.json"))
    img_files = list(mapped_path.glob("*.png"))


    # Läs in data från data_mapped
    mapped_dict = {"lat": [], "lon": [], "path": []}
    for i in range(len(json_files)):
        assert json_files[i].stem == img_files[i].stem, f"Filerna {json_files[i]} och {img_files[i]} matchar inte!"
        # Läs in JSON-data
        with open(json_files[i], encoding="utf-8") as f:
            data = json.load(f)
        df_mapped = pd.DataFrame()
        mapped_dict["lat"].append(data['coordinates'][0])
        mapped_dict["lon"].append(data['coordinates'][1])
        mapped_dict["path"].append(str(f"{MAPPED_PATH}/{img_files[i].stem}.png"))

    df_mapped = pd.DataFrame(mapped_dict)



    # Läs in data fron streetview
    df1 = pd.read_csv(streetview_coords_path, names=["lat", "lon"])

    for p in df1.index:
        jpg_path = convert_png_to_jpg(f"{streetview_path}/{str(p)}.png", f"{STREETVIEW_JPG_PATH}/{str(p)}.jpg")
        df1.at[p, "path"] = jpg_path


    df_cities = [df_mapped, df1]

    print(f"Totalt antal bilder i Street View: {len(df1)}")

    # Läs in data från cities
    for p in csv_files:
        try:
            df_city = pd.read_csv(p, usecols=["lat", "lon", "panoid"])
            df_city = handle_dataset_cities(df_city, p, cities_image_path)
            df_city = df_city.drop("panoid", axis=1)
            df_city_sampled = df_city.sample(min(len(df_city), min(10000, len(df1) // len(csv_files))), random_state=1)
            df_cities.append(df_city_sampled)

        except Exception as e:
            print(f"Varnar: kunde inte läsa {p} ({e})")

    df = pd.concat(df_cities, ignore_index=True)
    print(f"Antal bilder i cities: {len(df) - len(df1)}")

    # Rensa upp: kasta rader utan lat/lon och utanför giltiga intervall
    df = df.dropna(subset=["lat", "lon"])
    df = df[(df["lat"] >= -90) & (df["lat"] <= 90) & (df["lon"] >= -180) & (df["lon"] <= 180)]

    LEVEL = 6  # justera för magnifikation
    def latlon_to_s2(lat, lon, level):
            return CellId.from_lat_lng(LatLng.from_degrees(lat, lon)).parent(level)
    for i in range(LEVEL):
        # Mappa varje punkt till sin S2-cell på vald nivå
        df[f"cell_id_{i+1}"] = [latlon_to_s2(lat, lon, i+1).id() for lat, lon in zip(df.lat, df.lon)]

    df.to_csv("data.csv", index=False)
    return df



def convert_png_to_jpg(png_path, jpg_path):
    path = Path(jpg_path)
    path.parent.mkdir(parents=True, exist_ok=True)


    im = Image.open(png_path)
    rgb_im = im.convert('RGB')
    rgb_im.save(jpg_path)

    return jpg_path


def handle_dataset_cities(df, city_csv_path, cities_image_path) -> pd.DataFrame: #This is just to deal with the cities dataset
    """Add correct image path for each row in a city's dataframe."""
    city_name = Path(city_csv_path).stem  

    img_folder_path = Path(f"{cities_image_path}/{city_name}")

    img_list = list(img_folder_path.glob("*.jpg"))

    img_panoid_list = [image.stem[-22:] for image in img_list]

    for index in df.index:
        if df.at[index, "panoid"] in img_panoid_list:
            df.at[index, "path"] = f"./cities/Images/{city_name}/{img_list[img_panoid_list.index(df.at[index, 'panoid'])].name}"
        else:
            df.at[index, "path"] = None
    print(f"Totalt antal bilder i {city_name}: {len(img_list):,}")
    return df

