import pickle
from sklearn.metrics import classification_report

def test_model(testing_data_file, model_file='Data/model.pkl'):
    with open(testing_data_file, 'rb') as f:
        MoveData_test, MoveType_test = pickle.load(f)
    with open(model_file, 'rb') as f:
        clf = pickle.load(f)
    MoveType_pred = clf.predict(MoveData_test)
    # print(MoveType_pred, '\n', MoveType_test)
    print("=== Wyniki ===")
    print(classification_report(MoveType_test, MoveType_pred))