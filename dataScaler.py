from sklearn.preprocessing import StandardScaler
from dataAnalyser import plot_histograms_from_array


def scale_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # print_scaled_comparison(X, X_scaled)

    return X_scaled


def print_scaled_comparison(X, X_scaled):
    index_of_mcs0 = X.columns.get_loc('mcs0')
    index_of_mcs1 = X.columns.get_loc('mcs1')
    print(index_of_mcs0)
    print(index_of_mcs1)

    print("Data before scale:")
    print("\n")
    print("mcs0:")
    print(X['mcs0'].head(10))
    print("\n")

    print("mcs1:")
    print(X['mcs1'].head(10))
    print("\n")
    print(X)
    print("\n")

    print("Data after scale:")
    print("\n")
    print("mcs0:")
    scaled_msc0_column = X_scaled[:, 1]
    print(scaled_msc0_column)
    plot_histograms_from_array(scaled_msc0_column, index_of_mcs0)

    print("mcs1:")
    scaled_msc1_column = X_scaled[:, 2]
    print(scaled_msc1_column)
    plot_histograms_from_array(scaled_msc0_column, index_of_mcs1)
    print("\n")
    print(X_scaled)
    print("\n")

    for index in range(X_scaled.shape[1]):
        column_to_plot_histogram = X_scaled[:, index]
        plot_histograms_from_array(column_to_plot_histogram, index)

