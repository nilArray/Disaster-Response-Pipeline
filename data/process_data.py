import sys
import pandas as pd
from sqlalchemy import create_engine

def convert_misencoded(cell):
    '''
    Remove misencoded cells. Some cells have a number besides zero or one, this converts them to zero.
    
    Parameters:
    cell (int): A pandas dataframe cell with numeric data
    
    Returns:
    int: A variable we expect (0 or 1) or 0
    '''
    if cell > 1 or cell < 0:
        return(0)
    else:
        return(cell)

def extract_category(cell):
    '''
    Splits text on spaces and returns the first element. Intended to be used in a pd.apply()
    For example, a cell containing 'hello 15' would return 'hello'
    
    Parameters:
    cell (str): A pandas dataframe cell with text data
    
    Returns:
    str: first part of a cell separated by space
    '''
    split_cell = cell.split()
    category = split_cell[0]
    
    return(category)

def extract_value(cell):
    '''
    Splits text on spaces and returns the second element. Intended to be used in a pd.apply()
    For example, a cell containing 'hello 15' would return '15'
    
    Parameters:
    cell (str): A pandas dataframe cell with text data
    
    Returns:
    str: second part of a cell separated by space
    '''
    split_cell = cell.split()
    value = split_cell[1]
    
    return(value)

def load_data(messages_filepath, categories_filepath):
    '''
    Loads messages and categories dataset.
    
    Parameters:
    messages_filepath (str): The path to a CSV file containing message data
    categories_filepath (str): The path to a CSV file containing category data associated with messages
    
    Returns:
    pandas DataFrame: A DataFrame with both datasets combined
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = pd.merge(messages, categories, on = 'id')
    return(df)

def clean_data(df):
    '''
    Takes existing dataset and prepares it for ML.
    
    WARNING: Data must be in a very particular format or this step will fail.
    
    Parameters:
    df (pandas DataFrame): The DataFrame to clean
    
    Returns:
    pandas DataFrame: A cleaned DataFrame set up for easy ML
    '''
    # Pull out categories into their own dataframe
    categories = df['categories'].str.split(';', expand = True)

    # Iterate over each column and remove hyphens
    columns = categories.columns
    for column in columns:
        categories[column] = categories[column].str.replace('-', ' ')
        
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(extract_category).tolist()
    categories.columns = category_colnames

    # Remove the text portion and just leave the number
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(extract_value)
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
        # convert misencoded variables to correct ones
        categories[column] = categories[column].apply(convert_misencoded)
              
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis = 1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    df = df.drop_duplicates()
    
    return(df)

def save_data(df, database_filename):
    '''
    Save data to SQLite instance.
    
    Parameters:
    df (pandas DataFrame): The DataFrame intended to be converted to SQLite
    database_filename (str): Path to where the resultant SQLite db will be saved
    '''
    engine = create_engine('sqlite:///' + str(database_filename))
    df.to_sql('InsertTableName', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()