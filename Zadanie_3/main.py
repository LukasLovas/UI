import sklearn.preprocessing
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sklearn.datasets as datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Classification_model import Classification_model


def load_dataset():
    ds = datasets.load_iris()
    dataframe = pd.DataFrame(data=ds.data, columns=ds.feature_names)
    dataframe['species'] = ds.target
    print("-----------------------------")
    print("Hlavička Dataframe-u: ")
    print(dataframe.head())
    print("-----------------------------")
    print("Overenie nulových hodnôt v Dataframe: ")
    print(dataframe.isnull().sum())
    print("-----------------------------")
    print("Rozmery Dataframe-u: ")
    print(dataframe.shape)

    return dataframe


def scale_and_edit_dataset(dataframe):
    scaler = StandardScaler()
    x = dataframe.copy().drop(["species"], axis=1).values   #Data for classification
    y = dataframe["species"].values                         #Classification category
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_train = torch.FloatTensor(x_train)
    x_test = torch.FloatTensor(x_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    df = load_dataset()
    x_train, x_test, y_train, y_test = scale_and_edit_dataset(df)
    input_features = len(df.drop("species", axis=1).columns)
    output_features = len(df["species"].unique())
    model = Classification_model(input_features, output_features)


