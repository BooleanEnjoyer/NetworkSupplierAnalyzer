import pandas as pd
from dataAnalyser import analyze_data, base_data_info


def get_download_data():
    o2_download = pd.read_csv("C:\\Users\\Dominik\\Desktop\\Campaign3\\o2_download_nexus5x.csv")
    telekom_download = pd.read_csv("C:\\Users\\Dominik\\Desktop\\Campaign3\\telekom_download_nexus5x.csv")
    vodafone_download = pd.read_csv("C:\\Users\\Dominik\\Desktop\\Campaign3\\vodafone_download_nexus5x.csv")
    download_data = [o2_download, telekom_download, vodafone_download]
    return download_data


def get_upload_data():
    o2_upload = pd.read_csv("C:\\Users\\Dominik\\Desktop\\Campaign3\\o2_upload_nexus5x.csv")
    telekom_upload = pd.read_csv("C:\\Users\\Dominik\\Desktop\\Campaign3\\telekom_upload_nexus5x.csv")
    vodafone_upload = pd.read_csv("C:\\Users\\Dominik\\Desktop\\Campaign3\\vodafone_upload_nexus5x.csv")
    upload_data = [o2_upload, telekom_upload, vodafone_upload]
    return upload_data


def get_company_names():
    company_names = [
        "o2",
        "telekom",
        "vodafone"
    ]
    return company_names


def process_data(data, current_data_type):
    # base_data_info(data)

    if current_data_type == 'Download':
        mapped_data = map_modulations_to_numbers(data)
        # analyze_data(mapped_data, current_data_type)
        X = mapped_data.drop(['throughput', 'tp_cleaned', 'chipsettime', 'gpstime', 'longitude', 'latitude', 'speed'], axis=1)
        y = mapped_data['throughput']
    if current_data_type == 'Upload':
        # analyze_data(data, current_data_type)
        X = data.drop(['qualitytimestamp', 'tp_cleaned', 'gpstime', 'longitude', 'latitude', 'speed'], axis=1)
        y = data['tp_cleaned']
    print(f"Actual value min: {min(y)}")
    print(f"Actual value max: {max(y)}")
    return X, y


def map_modulations_to_numbers(data):
    mapping = {'QPSK': 1, '16QAM': 2, '64QAM': 3}
    data_to_map = data
    data_to_map['mcs0'] = data_to_map['mcs0'].map(mapping)
    data_to_map['mcs1'] = data_to_map['mcs1'].map(mapping)
    return data_to_map
