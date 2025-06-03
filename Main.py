from ProcessData import process_dataset
from TrainModel import train_model
from TestModel import test_model

def main():
    data_path = "./DataSet"
    process_dataset(data_path)
    train_model()
    test_model()

main()