import matplotlib.pyplot as plt
import pandas as pd
import sys

def build_scatter_plot(dataset):
    i = 6
    dataset = pd.read_csv(dataset)
    for i in range(6, 19):
        x = dataset.iloc[:, [1, i, i + 1]].values

        Gryffindor = [[],[]]
        Hufflepuff = [[],[]]
        Ravenclaw = [[],[]]
        Slytherin = [[],[]]

        for el in x:
            if  el[0] == "Gryffindor":
                Gryffindor[0].append(el[1])
                Gryffindor[1].append(el[2])
            elif el[0] == "Hufflepuff":
                Hufflepuff[0].append(el[1])
                Hufflepuff[1].append(el[2])
            elif el[0] == "Ravenclaw":
                Ravenclaw[0].append(el[1])
                Ravenclaw[1].append(el[2])
            elif el[0] == "Slytherin":
                Slytherin[0].append(el[1])
                Slytherin[1].append(el[2])
            else:
                return print("Invalid day of week")
        plot_scatter(Gryffindor, Hufflepuff, Ravenclaw, Slytherin, i, dataset)

def plot_scatter(Gryffindor, Hufflepuff, Ravenclaw, Slytherin, i, dataset):
    n_bins = 25

    plt.scatter(Gryffindor[0], Gryffindor[1], alpha=0.5)
    plt.scatter(Hufflepuff[0], Hufflepuff[1], alpha=0.5)
    plt.scatter(Ravenclaw[0], Ravenclaw[1], alpha=0.5)
    plt.scatter(Slytherin[0], Slytherin[1], alpha=0.5)
    plt.title(dataset.columns.values[i] + " " + dataset.columns.values[i+1])

    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Please provide just one dataset")
        exit(0)

    build_scatter_plot(sys.argv[1])

if __name__ == "__main__":
    main()
