import matplotlib.pyplot as plt
import pandas as pd
import sys

def build_histogram(dataset):
    dataset = pd.read_csv(dataset)
    for i in range(6, 19):
        x = dataset.iloc[:, [1, i]].values

        Gryffindor  = []
        Hufflepuff  = []
        Ravenclaw   = []
        Slytherin   = []

        for el in x:
            if  el[0] == "Gryffindor":
                Gryffindor.append(el[1])
            elif el[0] == "Hufflepuff":
                Hufflepuff.append(el[1])
            elif el[0] == "Ravenclaw":
                Ravenclaw.append(el[1])
            elif el[0] == "Slytherin":
                Slytherin.append(el[1])
            else:
                return print("WTF")
        plot_histogram(Gryffindor, Hufflepuff, Ravenclaw, Slytherin, i, dataset)

def plot_histogram(Gryffindor, Hufflepuff, Ravenclaw, Slytherin, i, dataset):
    n_bins = 100
    plt.hist(Gryffindor, bins=n_bins, alpha=0.5, label="Gryffindor", density=True)
    plt.hist(Hufflepuff, bins=n_bins, alpha=0.5, label="Hufflepuff", density=True)
    plt.hist(Ravenclaw, bins=n_bins, alpha=0.5, label="Ravenclaw", density=True)
    plt.hist(Slytherin, bins=n_bins, alpha=0.5, label="Slytherin", density=True)
    plt.title(dataset.columns.values[i])
    plt.legend(loc='upper right')
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Please provide just one dataset")
        exit(0)

    build_histogram(sys.argv[1])

if __name__ == "__main__":
    main()
