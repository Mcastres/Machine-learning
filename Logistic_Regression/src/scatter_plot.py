import matplotlib.pyplot as plt
import pandas as pd
import sys

def find_index(header):
    indexes = []
    i = 0
    for value in header:
        if value == sys.argv[2] or value == sys.argv[3]:
            indexes.append(i)
        i += 1
    if len(indexes) != 2:
        print("Something went wrong, please verify your subjects")
        exit(0)
    return indexes

def build_scatter_plot(dataset):
    i = 6
    dataset = pd.read_csv(dataset)

    indexes = find_index(dataset.columns.values)
    x = dataset.iloc[:, [1, indexes[0], indexes[1]]].values

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
    plot_scatter(Gryffindor, Hufflepuff, Ravenclaw, Slytherin, indexes, dataset)

def plot_scatter(Gryffindor, Hufflepuff, Ravenclaw, Slytherin, indexes, dataset):
    n_bins = 25

    plt.scatter(Gryffindor[0], Gryffindor[1], alpha=0.5, label="Gryffindor")
    plt.scatter(Hufflepuff[0], Hufflepuff[1], alpha=0.5, label="Hufflepuff")
    plt.scatter(Ravenclaw[0], Ravenclaw[1], alpha=0.5, label="Ravenclaw")
    plt.scatter(Slytherin[0], Slytherin[1], alpha=0.5, label="Slytherin")
    plt.title(dataset.columns.values[indexes[0]] + "(y) vs " + dataset.columns.values[indexes[1]] + "(x)")
    plt.show()

def main():
    if len(sys.argv) != 4:
        print("Please provide just one dataset and two subjects")
        exit(0)

    build_scatter_plot(sys.argv[1])

if __name__ == "__main__":
    main()
