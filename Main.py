"""
Plik main.
"""

from ProcessData import process_dataset
from LstmTrain import train_from_pkl
from LstmTest import test_from_pkl


def main():
    process_dataset("./TrainDataSet", "Data/trainData.pkl", "train")  # zbior treningowy
    train_from_pkl("Data/trainData.pkl", "Data/model.pth")
    process_dataset("./TestDataSet", "Data/testData.pkl", "test")  # zbior testowy
    test_from_pkl("Data/testData.pkl", "Data/model.pth")


main()
