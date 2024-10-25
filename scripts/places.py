import requests
import json
import csv
import logging as logger
from requests.structures import CaseInsensitiveDict
import os

def get_places(url, api_key):
    """
    Gets places based on type (restaurants, hotels etc) from urls generated from Geoapify (Egypt)
    https://apidocs.geoapify.com/playground/places/ 
    And dumps places data to data.json file
    """
    f_data = list()
    headers = CaseInsensitiveDict()
    headers["Accept"] = "application/json"
    # res = requests.get(f'https://api.geoapify.com/v1/geocode/search?country=Egypt&apiKey={os.getenv("GEO_API_KEY")}', headers=headers)
    try:
        res = requests.get("{}&apiKey={}".\
                        format(url, api_key), headers=headers)
        data = res.json()
        logger.info('Data Fetched')
        if data.get('error'):
            logger.error(f"An error occured while fetching data: \
                         {data.get('error')} - {data.get('message')}")
        else:
            with open('data.json', 'r') as f:
                f_data = json.load(f)
            f_data.append(data)
            with open('data.json', 'w') as f:
                json.dump(f_data, f)
    except Exception as e:
        logger.error(f'An Error occured {e}')
    return 'Success'   

def data_tocsv(json_file, csv_file, header):
    # json_file is data.json
    f_data = list()
    if json_file:
        with open(json_file, 'r') as f:
            f_data = json.load(f) # A list JSON data
        
        csv_file = open(csv_file, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        count = 0
        keys = list()
        for data in f_data:
            for entry in data['features']:
                if count == 0:
                    csv_writer.writerow(header)
                    count += 1
                for k, v in entry['properties'].items():
                    keys.append(k)
                for k in keys:
                    if k not in header:
                        if entry['properties'].get(k):
                            del entry['properties'][k]
                        else:
                            continue
                csv_writer.writerow(entry['properties'].values())
        csv_file.close()



    

if __name__ == '__main__':
    #print(get_places('https://api.geoapify.com/v2/places?categories=commercial.gift_and_souvenir,commercial.food_and_drink,entertainment.culture,entertainment.zoo,entertainment.museum,entertainment.cinema,entertainment.amusement_arcade,entertainment.theme_park,entertainment.activity_park,entertainment.water_park&filter=rect:30.977098153485088,26.24141530203768,35.41298616787856,22.031680620775376&limit=500', os.getenv('GEO_API_KEY')))
    data_tocsv('data.json', 'data.csv', ["name","state",
                                         "street","lon","lat",
                                         "formatted","categories"])