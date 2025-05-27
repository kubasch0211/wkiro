from process_data import process_dataset
from train_model import train_model
from test_model import test_model

def main():
    data_path = 'Data_Run_Walk'  # <- TUTAJ ustaw ścieżkę do danych
    process_dataset(data_path)
    train_model()
    test_model()

if __name__ == "__main__":
    main()
