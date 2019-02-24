import re, pickle, sys

import pandas as pd

from sqlalchemy import create_engine

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

def load_data(database_filepath):
    '''
    Loads data from the passed in path via SQLite
    
    Parameters: 
    database_filepath (str): The path to a supported database. It must have a table titled 'InsertTableName'
    
    Returns:
    series: The X input text to be analyzed
    dataframe: The one-hot encoded target Y values
    list: The names of the columns or 'categories'
    '''
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('InsertTableName', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre', 'child_alone'], axis = 1)
    return(X.values, Y.values, Y.columns)


def tokenize(text):
    '''
    Cleans and tokenizes text
    
    Parameters:
    text (str): Arbitrary string to tokenize
    
    Returns:
    str: A string with numbers, symbols, stopwords removed and lemmatized 
    '''
    
    # Replace punctuation with spaces and make the case all the same
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    words = text.split()
    
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    # Reduce words
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]

    return(lemmed)


def build_model():
    '''
    Generates the pipeline and model
    
    Returns:
    GridSearchCV: A pipeline that countvectorizes, tfidfs, and contains a RandomForestClassifier. GridSearch parameters included to tune th model
    '''
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators' : [100],
        'clf__estimator__min_samples_split' : [2],
        'clf__estimator__min_samples_leaf': [1] 
    }
    
    optimized_model = GridSearchCV(pipeline, param_grid = parameters, n_jobs = -1, cv = 2)
    return(optimized_model)


def evaluate_model(model, X_test, Y_test, category_names):    
    '''
    Performs column based validation of a model. Used for multi-output classification. Category names should be in the same order 
    
    Paramters:
    model (BaseEnsemble): a trained sklearn model to evaluate
    X_test: tokenzied text used to test a model
    Y_test: one-hot encoded multi-output results to be be predicted with
    category
    '''
    Y_predict = model.predict(X_test)
    
    for index, column in enumerate(category_names):
        # Pretty print title
        print(column.title())
        print('----------------------------------------------------------')
        
        # Select the current column and all rows from test and predicted datasets for classification
        print(classification_report(Y_test[:, index], Y_predict[:, index]))
        print('')

def save_model(model, model_filepath):
    '''
    Save a trained model to the file system. This uses Python's pickling process to save the object's state.
    
    Parameters:
    model (BaseEnsamble): The trained model to save
    model_filepath (str): Path to where the pickled model will be saved
    '''
    with open(model_filepath, 'wb') as file_handle:
        pickle.dump(model, file_handle)

def main():
    print(sys.argv)
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()