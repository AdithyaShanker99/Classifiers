import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def load_data(url) :
    df = pd.read_csv(url)
    return df

def process_data(df) :
    df['variety'] = df['variety'].replace('Setosa', 0.0)
    df['variety'] = df['variety'].replace('Versicolor', 1.0)
    df['variety'] = df['variety'].replace('Virginica', 2.0)
    features = df.drop('variety', axis=1)
    output = df['variety']
    return features, output

def getTrainAndTest(features, output) :
    X_train, X_test, Y_train, Y_test = train_test_split(features, output, test_size=0.2, random_state=41)
    return X_train, X_test, Y_train, Y_test