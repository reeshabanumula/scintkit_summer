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

#start connecting key values to variables used in the code

Data_folder1 = config['Data_folder1']
Data_folder2 = config['Data_folder2']
thresh = float(config['Threshold for s4'])

Interpolate_for_nan= config['Interpolating for Nan Values'] == 'True' 
origin_loc = float(config['origin location'])
latitude = float(config['latitude'])
longitude = float(config['longitude'])
sat = config["satellite"]
R_earth = float(config['R_earth'])


#binzip conversion

input_pattern = config['input pattern']
input_root = config['input root']
output_root = config['output root']
temp_root = config['temp root']
verbose =config['verbose'] == 'True'



#need to finalize proper values for origin lat lon and sat
#need to develop interpolate or drop nan values solution

#R = 6371 #km