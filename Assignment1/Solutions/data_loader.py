import pandas as pd  # type: ignore

def data_loader(file_path:str, title:str) -> pd.DataFrame: 
    # Reading the .csv file
    df = pd.read_csv(file_path, header=None)
    print(f'\nDescription of the {title} data:\n',df.describe())

    # Naming columns for ease of use
    df.columns = ['x1', 'x2', 'y']
    print(f'\nColumns in {title} dataset:\n', df.columns)

    # Checking for Missing / NaN values
    if(not df.isnull().sum().all()): 
        print('\nNo Missing / NaN Values found!\n')
    return df