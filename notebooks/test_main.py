import pytest
from click.testing import CliRunner
from main import main
import os
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from tempfile import NamedTemporaryFile
from sklearn.model_selection import train_test_split

@pytest.fixture
def create_temp_csv():
    train_data = {
        'published_date': ["2020-12-12", "2020-10-12", "2020-12-10", "2021-12-12"],
        'published_platform': ["Desktop", "Desktop", "Desktop", "Desktop"],
        'rating': [2, 3, 4, 5], 
        'type': ["review", "review", "review", "review"],
        'text': ["Bad company", "It's OK", "Good company", "Perfect company!"],
        'title': ["Very disappointed !", "OK", "Good", "Perfect!"], 
        'helpful_votes': [0, 1, 0, 2]
    }
    temp_file = NamedTemporaryFile(delete=False, mode='w', newline='')

    pd.DataFrame(train_data).to_csv(temp_file.name, index=False)

    return temp_file.name


#train
def test_train_command(create_temp_csv):
    runner = CliRunner()
    train = create_temp_csv
    test = create_temp_csv
    result = runner.invoke(main, [
        "train", 
        "--data", train, 
        "--test", test, 
        "--model", "model2.pkl"
    ])
    assert result.exit_code == 0
    assert os.path.exists("model2.pkl")


#predict
def test_predict_command(create_temp_csv):
    runner = CliRunner()
    train = create_temp_csv
    test = create_temp_csv

    result = runner.invoke(main, [
        'predict', 
        '--model', "model2.pkl", 
        '--data', "Good company"
    ])

    assert result.exit_code == 0
    assert result.output.strip()[1] == '1'


#split
def test_data_split(create_temp_csv):
    runner = CliRunner()
    df = create_temp_csv
    data = pd.read_csv(df, header=None)
    train, test = train_test_split(df, test_size=0.25)
    assert len(test)==4
    assert len(train)==12