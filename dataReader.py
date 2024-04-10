import pandas as pd
from dataAnalyser import analyze_data, base_data_info


def get_data():
    data = pd.read_csv("C:\\Users\\Dominik\\Desktop\\Campaign3\\o2_download_nexus5x.csv")
    # base_data_info(data)
    mapped_data = map_modulations_to_numbers(data)
    # analyze_data(mapped_data)

    X = mapped_data.drop(['throughput', 'tp_cleaned', 'chipsettime', 'gpstime', 'longitude', 'latitude', 'speed'], axis=1)
    y = data['throughput']
    return X, y


def map_modulations_to_numbers(data):
    mapping = {'QPSK': 1, '16QAM': 2, '64QAM': 3}
    data_to_map = data
    data_to_map['mcs0'] = data_to_map['mcs0'].map(mapping)
    data_to_map['mcs1'] = data_to_map['mcs1'].map(mapping)
    return data_to_map
