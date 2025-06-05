from ProcessData import process_dataset
from TrainModel import train_model
from TestModel import test_model

def main():
    process_dataset("./TrainDataSet", "Data/trainData.pkl")  # training set
    process_dataset("./TestDataSet", "Data/testData.pkl")  # testing set
    train_model("Data/trainData.pkl")
    test_model("Data/testData.pkl")

main()