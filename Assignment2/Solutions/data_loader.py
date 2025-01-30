# -------------------------------------------- Data Loading --------------------------------------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA5400 Foundations of Machine Learning Assignment 2
#
#   	-	This file contains code for checking NaN / Duplicate values in the dataset and removing them. 
#   	-	There are also functions which split the parent .csv dataset into training and testing datasets 
#   	-	To faciliate testing for the given Spam Classification problem, there are functions to create 
#			text files from .csv text file. 
#
# --------------------------------------------------------------------------------------------------------------

import os 
import pandas as pd 

def clean_dataframe(dataset: pd.DataFrame) -> pd.DataFrame: 
	null_values = dataset.isnull().sum()
	na_values = dataset.isna().sum()

	if(null_values.any() or na_values.any()):
		dataset = dataset.dropna() 
	if(dataset.duplicated().any): 
		dataset = dataset.drop_duplicates()

	return dataset


def split_csv_file(input_file: pd.DataFrame, output_file1_name:str, output_file2_name:str, split_ratio:float = 0.5) -> None:
    # Shuffle the DataFrame rows
    df_shuffled = input_file.sample(frac = 1, random_state = 42).reset_index(drop = True)
    
    # Calculate the split index
    split_index = int(len(df_shuffled) * split_ratio)
    
    # Split the DataFrame into two
    df1 = df_shuffled.iloc[:split_index]
    df2 = df_shuffled.iloc[split_index:]
    
	# Adding file extension '.csv' 
    output_file1_name += '.csv'
    output_file2_name += '.csv'

    # Save the two new DataFrames to separate CSV files
    df1.to_csv(output_file1_name, index = False)
    df2.to_csv(output_file2_name, index = False)

def save_messages_to_files(df: pd.DataFrame, output_folder='test') -> None:

    # Create the output folder if it doesn't already exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over each row in the DataFrame and save the messages as text files
    for index, row in df.iterrows():
        message = row['Message']
        filename = f'email_{index + 1}.txt'
        filepath = os.path.join(output_folder, filename)
        
        with open(filepath, 'w') as file:
            file.write(message)

    print(f"\nMessages have been saved as text files in the '{output_folder}' folder.\n")


def read_emails(directory_path:str, vectoriser, model) -> None:
    # Check if the folder exists
    if not os.path.exists(directory_path):
        print(f"The folder '{directory_path}' does not exist.")
        return

    # List all files in the folder
    files = os.listdir(directory_path)
    
    # Filter out files that are named 'email_x.txt'
    email_files = [file for file in files if file.startswith('email_') and file.endswith('.txt')]
    
    # Sort the files to ensure they are read in order
    email_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    # Read and print the contents of each email file
    for email_file in email_files:
        file_path = os.path.join(directory_path, email_file)
        with open(file_path, 'r') as file:
            content = file.read()
            #print(f"Content of {email_file}:\n{content}\n{'-'*40}")
            content_text = [content]
            content_vector = vectoriser.transform(content_text)
            if(model.predict(content_vector) == 0): 
                print(f'Prediction:\t{model.predict(content_vector)}\tHam Email\n')
            elif(model.predict(content_vector) == 1):
                print(f'Prediction:\t{model.predict(content_vector)}\tSpam Email\n')

def count_files_in_folder(folder_path: str) -> int:
    try:
        # List all files and directories in the specified folder
        files_and_dirs = os.listdir(folder_path)
        
        # Filter out directories, keeping only files
        files = [f for f in files_and_dirs if os.path.isfile(os.path.join(folder_path, f))]
        
        # Return the count of files
        return len(files)
    
    except FileNotFoundError:
        print(f"The folder '{folder_path}' does not exist.")
        return 0

