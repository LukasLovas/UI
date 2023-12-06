import torch
import numpy as np
import torch.nn as nn
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
    x = dataframe.copy().drop(["species"], axis=1).values  # Data for classification
    y = dataframe["species"].values  # Classification category
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_train = scaler.fit_transform(x_train)

    x_test = scaler.transform(x_test)
    x_train = torch.FloatTensor(x_train)
    x_test = torch.FloatTensor(x_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    return x_train, x_test, y_train, y_test


def training(epochs):
    loss_list = np.zeros(epochs)
    accuracy_list = np.zeros(epochs)

    for epoch in range(epochs):
        optimizer.zero_grad()  # Delete all gradient from each of the optimized tensors
        model_output_train = model(x_train)  # Forwarding the training data
        loss_train = loss_criterion(model_output_train, y_train)  # Compare model output data with real result data
        loss_train.backward()  # Calculate gradients
        optimizer.step() #Update weights
        loss_list[epoch] = loss_train.item()

        with torch.no_grad():
            y_predicted = model(x_train)
            correct = (torch.argmax(y_predicted, dim=1) == y_test).type(torch.FloatTensor)
            accuracy_list[epoch] = correct.mean()

    visualize_result(loss_list, accuracy_list)


def visualize_result(loss_list, accuracy_list):
    plt.figure(figsize=(10, 10))
    plt.plot(loss_list, label="Training loss")
    plt.plot(accuracy_list, label="Test accurate hits")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    df = load_dataset()
    x_train, x_test, y_train, y_test = scale_and_edit_dataset(df)
    input_features = len(df.drop("species", axis=1).columns)
    output_features = len(df["species"].unique())
    model = Classification_model(input_features, output_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_criterion = nn.CrossEntropyLoss()
    losses = training(100)
