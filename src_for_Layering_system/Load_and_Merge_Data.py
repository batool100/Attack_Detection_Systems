
import pandas as pd
import os



def load_and_merge_files(folder_path):
    """
    Load all CSV files from a folder and merge them
    into a single DataFrame.
    """

    # List to store the data
    all_dfs = []

    # Read each CSV file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            all_dfs.append(df)

    # Combine all the data into a single DataFrame
    df = pd.concat(all_dfs, ignore_index=True)

    # Display basic information about the data which number of rows and columns
    print("Dataset shape:", df.shape)

    return df

