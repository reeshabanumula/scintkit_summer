import pandas as pd
from pathlib import Path

#create path from txt file to py

config_path = Path(__file__).parent / "configurations.txt"

with open(config_path, "r") as file:
    lines = file.readlines()

config = {}

for line in lines:
    if line.strip() == '' or line.startswith('#'):
        continue
    key, value = line.split('=', 1)
    key = key.strip()
    value = value.strip()

    config[key] = value

#file locations
Data_folder1 = config['Data_folder1']
Data_folder2 = config['Data_folder2']
input_directory = (config['input_directory'])


#analysis details
thresh = float(config['s4_threshold'])
elevation_filter = float(config['elevation_filter'])


r_latitude = float(config['receiver_latitude'])
r_longitude = float(config['receiver_longitude'])
r_height = float(config['reciever_height'])
lat_tol = float(config['latitude_tolerance'])
lon_tol = float(config['longitude_tolerance'])

r_B_latitude = float(config['receiver_B_latitude'])
r_B_longitude = float(config['receiver_B_longitude'])
r_B_height = float(config['receiver_B_height'])

max_workers = int(config['parallel_process_max_workers'])


R_earth = float(config['earth_radius_km'])

#sat = config["satellite"]
nan_method = config["nan_method"]


#binzip conversion
input_pattern = config['input_pattern']
input_root = config['input_root']
output_root = config['output_root']
temp_root = config['temp_root']
verbose =config['verbose'] == 'True'


#output folders
output_folder = config["output_folder"]
cross_correlation_file = config['cross_correlation_file']


# File pairing options
pairing_mode = config["pairing_mode"].strip().lower()
pairing_tolerance = int(config["pairing_tolerance"])
unpaired_file_action = config["unpaired_file_action"].strip().lower()