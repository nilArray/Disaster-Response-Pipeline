import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def cold_truth_table(row):
    '''
    Converts a cold and shelter 'truth table' to a understandable format.
    
    Parameters:
    row (Pandas DataFrame Row): apply(axis = 1) to a truth table dataframe
    
    Returns:
    A sensible description of the truth table
    '''
    
    cold = bool(row['cold'])
    shelter = bool(row['shelter'])
    
    if cold and shelter:
        return('Shelter and Cold')
    
    if cold and not shelter:
        return('Just Cold')
    
    if not cold and shelter:
        return('Just Shelter')
    
    if not cold and not shelter:
        return('Other Issue')
    
    return('error')

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/MyDb.db')
df = pd.read_sql_table('InsertTableName', engine)

# load model
model = joblib.load("../models/model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # get data for cold/shelter ratios
    cold_ratio = df.groupby(['cold', 'shelter']).count()['id'].reset_index(drop = False)
    cold_ratio.columns = ['cold', 'shelter', 'n']
    cold_ratio['status'] = cold_ratio.apply(cold_truth_table, axis = 1)
    cold_ratio['ratio'] = round((cold_ratio['n'] / cold_ratio['n'].sum()) * 100, 2)
    
    # get data to calculate the number of issues people are reporting
    hist_columns = df.columns.tolist()[5:]
    df_sums = df.copy(deep = True)
    df_sums['Num_Issues'] = df_sums[hist_columns].sum(axis = 1)
   
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x = cold_ratio['status'],
                    y = cold_ratio['ratio']
                )
            ],

            'layout': {
                'title': 'Ratio of Messages With Cold and Shelter Classifications',
                'yaxis': {
                    'title': "Ratio"
                },
                'xaxis': {
                    'title': "Cold Status"
                },
                'annotations': {
                    
                }
            }
        },
        {
            'data': [
                Histogram(
                    x = df_sums['Num_Issues']
                )
            ],

            'layout': {
                'title': 'Distribution of Number of Issues',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Number of Issues"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()