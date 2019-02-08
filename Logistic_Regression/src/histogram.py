import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
import sys

enumerate 

def main():
    if len(sys.argv) != 2:
        print("Please provide just one dataset")
        exit(0)
    # feature = int(sys.argv[2])
    for i in range(6, 19):
        dataset = pd.read_csv(sys.argv[1])
        x = dataset.iloc[:, [1, i]].values
        # y = dataset.iloc[:, 1].values
        # imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
        # imputer = imputer.fit(x)
        # x = imputer.transform(x)
        HogwartsHouse1 = []
        HogwartsHouse2 = []
        HogwartsHouse3 = []
        HogwartsHouse4 = []
        for el in x:
            if  el[0] == "Gryffindor":
                HogwartsHouse1.append(el[1])
            elif el[0] == "Hufflepuff":
                HogwartsHouse2.append(el[1])
            elif el[0] == "Ravenclaw":
                HogwartsHouse3.append(el[1])
            elif el[0] == "Slytherin":
                HogwartsHouse4.append(el[1])
            else:
                return print("Invalid day of week")

        n_bins = 25
        plt.hist(HogwartsHouse1, bins=n_bins, alpha=0.5, label="Gryffindor")
        plt.hist(HogwartsHouse2, bins=n_bins, alpha=0.5, label="Hufflepuff")
        plt.hist(HogwartsHouse3, bins=n_bins, alpha=0.5, label="Ravenclaw")
        plt.hist(HogwartsHouse4, bins=n_bins, alpha=0.5, label="Slytherin")
        plt.title(dataset.columns.values[i])
        plt.legend(loc='upper right')
        print(i)
        plt.show()

if __name__ == "__main__":
    main()
