import pandas as pd


def get_data():
    data = pd.read_csv("C:\\Users\\Dominik\\Desktop\\Campaign3\\o2_download_nexus5x.csv")
    # data['category'] = data['quality'] >= 7
    data_head = data.head(20)
    print(data_head)
    print(data.dtypes)
    print(data.isnull().any())
    # X = data[data.columns[0:11]].values
    # y = data['category'].values.astype('int')
    # return X, y