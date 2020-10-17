import ssl
import urllib.request
import pandas as pd
import json
import os
from tqdm import tqdm
import argparse


def get_file_name(file_url):
    return str.split(file_url, '/')[-1]


def download_file(file_url: str):
    context = ssl._create_unverified_context()
    response = urllib.request.urlopen(file_url, context=context)
    return response.read()


def download_and_save_file(save_path: str, file_url: str):
    file_name = get_file_name(file_url)
    response = download_file(file_url)
    file = os.path.join(save_path, file_name)
    with open(file, 'wb') as f:
        f.write(response)
        f.close()
    return file


def radar_day_url_builder(day: int, month: int, year: int) -> str:
    return 'https://opendata-download-radar.smhi.se/api/version/latest/area/sweden/product/comp/' + str(
        year) + '/' + str(month) + '/' + str(day)


def download_radar_data(save_path: str, start_date: pd.datetime, end_date: pd.datetime, format='tif') -> list:
    file_list = []
    date_list = pd.date_range(start_date, end_date).tolist()
    for date in date_list:
        print(date)
        url = radar_day_url_builder(date.day, date.month, date.year)
        url_data = download_file(url)
        data_dict = json.loads(url_data)
        for d in data_dict.get('files'):  # look over date files
            df = [f for f in d.get('formats') if f.get('key') == format]  # filter data format to tif
            if len(df) == 1:  # if filter data length is one (only support tif format)
                file_url = df[0].get('link')
                file_name = get_file_name(file_url)
                file = os.path.join(save_path, file_name)
                if not os.path.isfile(file):
                    download_and_save_file(save_path, file_url)
                print("Downloaded file:" + file)
                file_list.append(file)
            elif len(df) == 0:
                print("Cant find format:" + format)
            else:
                raise Exception('data format length must be of size one')
        pass
    return file_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Data from SHMI')
    parser.add_argument('--start_day', type=int, default=1,
                        help='an integer for the accumulator')
    parser.add_argument('--end_day', type=int, default=31,
                        help='an integer for the accumulator')
    parser.add_argument('--start_month', type=int, default=1,
                        help='an integer for the accumulator')
    parser.add_argument('--end_month', type=int, default=12,
                        help='an integer for the accumulator')
    parser.add_argument('--start_year', type=int,
                        help='an integer for the accumulator')
    parser.add_argument('--end_year', type=int,
                        help='an integer for the accumulator')
    parser.add_argument('--save_folder', type=str,
                        help='an integer for the accumulator')

    args = parser.parse_args()
    start_time = pd.datetime(year=args.start_year, month=args.start_month, day=args.start_day)
    end_time = pd.datetime(year=args.end_year, month=args.end_month, day=args.end_day)
    download_radar_data(args.save_folder, start_time, end_time)
