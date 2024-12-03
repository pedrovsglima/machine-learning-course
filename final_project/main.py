import toml

from src.data import train_test_split


with open("config.toml", "r") as f:
    config = toml.load(f)


def main():

    # read files and split data into training and test data
    X_train, X_test, y_train, y_test = train_test_split.split_data_from_file(
        file_data=config["dataset"],
        test_size=config["dataset"]["test_size"],
        random_state=config["dataset"]["random_state"]
    )

    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)


if __name__ == "__main__":
    main()
