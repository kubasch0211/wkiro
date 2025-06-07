from process_data import process_dataset
from lstm_train import train_from_pkl
from lstm_test import test_from_pkl

def main():
    process_dataset("./TrainDataSet", "Data/trainData.pkl")  # training set
    train_from_pkl("Data/trainData.pkl", "model.pth")
    process_dataset("./TestDataSet", "Data/testData.pkl")  # testing set
    test_from_pkl("Data/testData.pkl", "model.pth")

main()