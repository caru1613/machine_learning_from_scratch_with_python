import numpy as np
import pandas as pd


def get_rating_matrix(filename, dtype=np.float32):
    df_data = pd.read_csv(filename)
    print(df_data)
    print(df_data["source"])

    # `    df_data.index = sorted(df_data["source"])
    #    print(df_data)

    #    del df_data["source"]
    #    print(df_data)

    pass


def get_frequent_matrix(filename, dtype=np.float32):
    pass


def main():
    get_rating_matrix("../movie_rating.csv")


if __name__ == "__main__":
    main()

