import train as gd

def get_thetas_from_file():
    try:
        file = open("Datasets/thetas")
        data = file.read().split(",")
        thetas = [float(data[0]), float(data[1])]
    except ValueError:
        print("Thetas as not correct, please try to train your model")
        exit(0)
    except FileNotFoundError:
        print("Datasets/thetas: No such file")
        exit(0)
    return (thetas)

def display_estimation(thetas, data):
    mileage = input("Please enter a mileage: ")
    while not mileage.isnumeric():
        input("Please provide a number: ")
    print("For a car with", mileage, "mileage, we estimate a price of:", round(gd.hypothesis(float(thetas[0]), float(thetas[1]), int(mileage))), "dollars!")

if __name__ == '__main__':
    thetas = get_thetas_from_file()
    data = gd.read_csv_file("Datasets/data.csv")
    display_estimation(thetas, data)
