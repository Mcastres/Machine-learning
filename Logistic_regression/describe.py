import sys
import csv
import numpy as np

def csv_to_data(file):
    data = np.genfromtxt(file, delimiter=',')
    return data

def describe_data(data):
    i               = 1
    n               = len(data[0])
    length          = []
    mean            = []
    std             = []
    min_val         = []
    first_quartile  = []
    second_quartile = []
    third_quartile  = []
    max_val         = []

    for i in range(1, n):
        column = data[:, [i]].ravel()
        if not (np.isnan(column[0])):
            length.append(len(column))
            mean.append(np.nansum(column) / length)
            std.append(np.nanstd(column))
            min_val.append(min(column))
            first_quartile, second_quartile, third_quartile.append(np.nanpercentile(column, [25, 50, 75]))
            max_val.append(max(column))

def main():
    if len(sys.argv) != 2:
        print("Please provide just one dataset")
        exit(0)

    data = csv_to_data(sys.argv[1])
    # print(data[:, [1, 3]])

    # Describe data
    describe_data(data)

if __name__ == "__main__":
    main()
