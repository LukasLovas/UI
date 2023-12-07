import datetime
import os
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import seaborn as sns
import sklearn.datasets as datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import tqdm
import matplotlib.pyplot as plt

from Feedforward_model_mix import Feedforward_model_mix
from Feedforward_model_relu import Feedforward_model_relu
from Recurrent_model import Recurrent_model


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


def train_feedforward(epochs):
    loss_list = np.zeros(epochs)
    predictions = []

    for epoch in tqdm.trange(epochs):
        optimizer.zero_grad()  # Delete all gradient from each of the optimized tensors - gradient is a vector that
        # signalizes which parameters are to be updated based on the loss criterion
        model_output_train = model(x_train)  # Forwarding the training data
        loss_train = loss_criterion(model_output_train, y_train)  # Compare model output data with real result data
        loss_train.backward()  # Calculate gradients
        optimizer.step()  # Update parameters - neural network learning
        loss_list[epoch] = loss_train.item()

    visualize_training(loss_list)


def train_rnn(epochs):
    loss_list = np.zeros(epochs)

    for epoch in tqdm.trange(epochs):
        optimizer.zero_grad()
        model_output_train = model(x_train.unsqueeze(1))
        loss_train = loss_criterion(model_output_train, y_train)
        loss_train.backward()
        optimizer.step()
        loss_list[epoch] = loss_train.item()

    visualize_training(loss_list)


def test_feedforward():
    model.eval()
    with torch.no_grad():
        predictions = []
        for value in x_test:
            value = value.unsqueeze(0)
            y_predicted = model.forward(value)
            predictions.append(y_predicted.argmax().item())

    accuracy = accuracy_score(y_test, predictions)
    print(f'{model.__class__.__name__} Accuracy: {accuracy * 100:.2f}%')
    visualize_testing(predictions)


def test_rnn():
    model.eval()
    with torch.no_grad():
        model_output_test = model(x_test.unsqueeze(1))
        _, predictions = torch.max(model_output_test, 1)

    accuracy = accuracy_score(y_test, predictions)
    print(f'{model.__class__.__name__} Accuracy: {accuracy * 100:.2f}%')
    visualize_testing(predictions)


def visualize_testing(predictions):
    # Save file
    output_dir = os.path.join("output", current_time)
    model_name = model.__class__.__name__
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, model_name)
    os.makedirs(path, exist_ok=True)

    # Accuracy table
    test_data_df = pd.DataFrame({'Real data': y_test, 'Predicted data': predictions})
    test_data_df['Correct'] = [1 if corr == pred else 0 for corr, pred in
                               zip(test_data_df['Real data'], test_data_df['Predicted data'])]
    plt.figure(figsize=(8, 6))
    plt.axis('off')
    tab = plt.table(cellText=test_data_df.values, colLabels=test_data_df.columns, cellLoc='center', loc='center', )
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(0.7, 0.7)
    plt.show()

    # Accuracy plot
    accuracy = accuracy_score(y_test, predictions)
    plt.figure(figsize=(8, 6))
    plt.plot(predictions, label='Indexes of data', marker='o', linestyle='None')
    plt.plot(y_test, label='Result labels', marker='x', linestyle='None')
    plt.title(f'Test Accuracy: {accuracy * 100:.2f}%')
    plt.xlabel('Indexes')
    plt.ylabel('Classes')
    plt.legend()
    plt.savefig(os.path.join(path, 'accuracy_plot.png'))
    plt.show()

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_test))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(path, 'confusion_matrix.png'))
    plt.show()


def visualize_training(loss_list):
    # Save file
    output_dir = os.path.join("output", current_time)
    model_name = model.__class__.__name__
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, model_name)
    os.makedirs(path, exist_ok=True)

    # Loss plot
    plt.figure(figsize=(10, 10))
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.plot(loss_list, label='Losses during training (difference between computed results and real results)')
    plt.legend()
    plt.savefig(os.path.join(path, 'loss_plot.png'))
    plt.show()


if __name__ == "__main__":
    df = load_dataset()
    x_train, x_test, y_train, y_test = scale_and_edit_dataset(df)
    input_features = len(df.drop("species", axis=1).columns)
    output_features = len(df["species"].unique())
    current_time = datetime.datetime.now().strftime("%d-%m-%Y %H.%M.%S")

    # FFNN_RELU
    model = Feedforward_model_relu(input_features, output_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_criterion = nn.CrossEntropyLoss()
    train_feedforward(100)
    test_feedforward()

    # FFNN_TANH_SOFTMAX
    model = Feedforward_model_mix(input_features, output_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_criterion = nn.CrossEntropyLoss()
    train_feedforward(100)
    test_feedforward()

    # RNN
    model = Recurrent_model(4, 64, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_criterion = nn.CrossEntropyLoss()
    train_rnn(100)
    test_rnn()
