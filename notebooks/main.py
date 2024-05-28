
import click
import os
import re
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
import nltk 
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 


nltk.download('punkt')
nltk.download('stopwords')

stemmer = PorterStemmer()
stopwords = set(stopwords.words('english'))

def preprocess_text(text):
    return re.sub(r"[^\w\s]+", '', text).lower().split()
    
def preprocess_sentence_eng(text):
    return ' '.join(map(stemmer.stem, preprocess_text(text)))
    
def delete_stopwords(text):
    preprocess_sentence_eng(text)
    tokens = word_tokenize(text) 
    filtered_sentence = [w for w in tokens if not w in stopwords] 
    return filtered_sentence

def preprocess_sentence_eng_2(text):
    return ' '.join(map(stemmer.stem, delete_stopwords(text)))

def preprocess_data(data):
    data['text_no_stopwords'] = data['text'].apply(preprocess_sentence_eng_2)
    data['rating_num'] = data['rating'].map({1: 0, 2: 0, 3: 0, 4: 1, 5: 1}) # переводим таргет в числа


@click.group()
def main():
    pass

@main.command()
@click.option('--data', required=True, help='Path to the training data file.')
@click.option('--test', help='Path to the test data file.')
@click.option('--split', type=float, help='split data into test and train set.')
@click.option('--model', required=True, help='Path to save the trained model.')
def train(data, test, split, model):
    model1 = Pipeline([("v", TfidfVectorizer()), ("m", LogisticRegression())])
    if data:
        if os.path.exists(data):
            df = pd.read_csv(data)
            preprocess_data(df)
        else: 
            raise FileNotFoundError("No such file. Write correct --data")
    if test:
        if os.path.exists(test):
            test_df = pd.read_csv(test)
            preprocess_data(test_df)
            x_train = df['text_no_stopwords']
            y_train = df['rating_num']
            x_test = test_df['text_no_stopwords']
            y_test = test_df['rating_num']
            
            model1.fit(x_train, y_train)
            y_pred = model1.predict(x_test)
            print(f1_score(y_pred, y_test))
        else:
            raise FileNotFoundError("No such file. Write correct --test file")
    elif split:
        train, test = train_test_split(df, test_size=split, random_state = 0)
        x_train = train['text_no_stopwords']
        x_test = test['text_no_stopwords']
        y_train = train['rating_num']
        y_test = test['rating_num']

        model1.fit(x_train, y_train)
        y_pred = model1.predict(x_test)
        print(f1_score(y_pred, y_test))
    else:
        click.echo("Either '--test' or '--split' must be provided.")
        return
        
    with open(model, 'wb') as f:
        pickle.dump(model1, f)

    click.echo("Model trained and saved successfully.")

@main.command()
@click.option('--model', required=True, help='Path to the trained model file.')
@click.option('--data', required=True, help='Path to the data file or text string.')
def predict(model, data):
    with open(model, 'rb') as f:
        model1 = pickle.load(f)

    if data.endswith('.csv'):
        df = pd.read_csv(data)
        preprocess_data(df)
        predictions = model1.predict(df)
        for pred in predictions:
            click.echo(pred)
    else:
        data = [preprocess_sentence_eng_2(data)]
        prediction = model1.predict(data)
        click.echo(prediction)

def preprocess_text(text):
    return text

if __name__ == "__main__":
    main()