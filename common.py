import torch
import json
import os
import datetime

PROJECT = 'RainMapGenerator'
M_CONDIONS = 2


def get_working_device():
    working_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Current Working Device is set to:" + str(working_device))
    return working_device


def datetime_folder_name():
    now = datetime.now()  # current date and time
    date_time = now.strftime("%d_%m_%Y_%H_%M_%S")
    return date_time


def init_folder(args):
    log_folder = os.path.join(args.log_folder, 'logs', datetime_folder_name())
    os.makedirs(log_folder, exist_ok=True)
    args_dict = vars(args)
    with open(os.path.join(log_folder, 'config.json'), 'w') as outfile:
        json.dump(args_dict, outfile)
    return log_folder
