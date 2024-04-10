import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_data(data):
    print()
    amount_of_rows_duplicated = data.duplicated().sum()
    print("Rows duplicated:")
    print(amount_of_rows_duplicated)
    print("\n")

    print("Amount of not unique throughput rows:")
    amount_of_not_unique_throughput_rows = data['throughput'].nunique()
    print(amount_of_not_unique_throughput_rows)
    print("\n")

    print("Mean throughput:")
    mean_throughput = data['throughput'].mean()
    print(mean_throughput)
    print("\n")

    # speed_info(data)

    # plot_histograms(data)
    # plot_scatterplot_of_throughput(data)
    plot_heat_map(data)


def base_data_info(data):
    rows_amount = 20
    data_head = data.head(rows_amount)
    print()
    print(f'First {rows_amount} rows of dataset:')
    print(data_head)
    print("\n")

    print("Dataset datatypes:")
    print(data.dtypes)
    print("\n")

    print("Has dataset null values in some column:")
    print(data.isnull().any())
    print("\n")


def speed_info(data):
    speed_array = np.array(data['speed'])
    amount_of_non_zeroes = np.count_nonzero(speed_array)
    print("Amount of zeroes in speed column:")
    print(amount_of_non_zeroes)
    print("\n")


def plot_histograms(data):
    for column in data.columns:
        data[column].hist()
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {column}')
        plt.show()


def plot_scatterplot_of_throughput(data):
    for column in data.columns:
        data.plot(y='throughput', x=column, kind='scatter')
        plt.xlabel(column)
        plt.ylabel('throughput')
        plt.title(f'Scatterplot of {column}')
        plt.show()


def plot_heat_map(data):
    corr = data.corr()
    plt.figure(figsize=[20,10])
    sns.heatmap(corr, annot=True)
    plt.xticks(rotation=45)
    plt.title("Heatmap of Correlation Coefficient", size=12)
    plt.show()


def plot_histograms_from_array(data, column_index):
    plt.hist(data)
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of column with index {column_index}')
    plt.show()
