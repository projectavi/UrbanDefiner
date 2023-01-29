import numpy as np
import pandas as pd
import os

# get the names of all the files in Urban folder under the data folder

def get_file_names():
    file_names = []
    for root, dirs, files in os.walk("data/Urban"):
        for file in files:
            if file.endswith(".csv"):
                file_names.append(os.path.join(root, file))
    return file_names

# load the data from the files and return a list of dataframes
def get_dataframes(file_names):
    dataframes = []
    for file_name in file_names:
        # want to use indices: 3 (word), 4 (def), 5 (eg)
        dataframes.append(pd.read_csv(file_name, usecols=[3, 4, 5], header = None))
    return dataframes

def merge(dfs):
    # merge all the dataframes into one
    return pd.concat(dfs)

def construct_data(df):
    # merge 2 columns in the dataframe with the word Definition: at the beginning and the word Example: in between
    df[1] = "Definition: " + df[1].astype(str) + " Example: " + df[2].astype(str)
    # drop the third column
    df.drop(columns=[2], inplace=True)
    # rename the columns
    df.rename(columns={0: "Word", 1: "Definition"}, inplace=True)
    # drop the rows with NaN values
    df.dropna(inplace=True)
    # drop the rows with empty strings
    df = df[df.Definition != ""]

    return df

def main():
    if not os.path.exists("./data/Urban/words.npy"):
        if (not(os.path.exists("./data/Urban/data.npy"))):
            file_names = get_file_names()
            dataframes = get_dataframes(file_names)
            dataframes = np.array(dataframes, dtype=object)
            np.save("./data/Urban/data.npy", dataframes)
        else:
            dataframes = np.load("./data/Urban/data.npy", allow_pickle=True)

        main_df = merge(list(dataframes))
        print(main_df.head())

        data_array = [[],[]]

        for index, row in main_df.iterrows():
            # write the word and definition to a file each on a new line
            # ", ".join()
            data_array[0].append(str(row[3]))
            temp = "Definition: " + str(row[4]) + ", Example: " + str(row[5])
            # remove all \ and whiever character is after it from string
            temp = ", ".join(temp.splitlines())
            data_array[1].append(temp)
        
        data_array = np.array(data_array, dtype=object)
        np.save("./data/Urban/words.npy", data_array)

    else:
        data_array = np.load("./data/Urban/words.npy", allow_pickle=True)
    
    print(data_array[0])


if __name__ == "__main__":
    main()