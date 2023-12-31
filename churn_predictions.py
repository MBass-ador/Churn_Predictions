import pandas as pd
from pycaret.classification import ClassificationExperiment


def load_data(filepath):
    # load churn data into DataFrame.
    dff = pd.read_csv(filepath, index_col='customerID')

    # drop engineered columns
    dff.drop(['AvgCharge', 'NormalizedAvgCharge'], axis='columns', inplace=True)

    return dff


def make_predictions(dfp):
    # use pycaret best model to make predictions on data in dataframe.
    classifier = ClassificationExperiment()

    model = classifier.load_model('churn_prediction_model')

    predict = classifier.predict_model(model, data=dfp)

    predict.rename({"prediction_score": "churn_probability"}, axis=1, inplace=True)

    return predict["churn_probability"]


if __name__ == "__main__":
    df = load_data('cleaned_churn_data.csv')

    predictions = make_predictions(df)

    print('Predictions:')
    print(predictions)
