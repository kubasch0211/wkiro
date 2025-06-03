import pickle
from sklearn.metrics import classification_report

def test_model(model_file='Data/model.pkl'):
    with open(model_file, 'rb') as f:
        clf, Coords_test, MoveType_test = pickle.load(f)
    MoveType_pred = clf.predict(Coords_test)
    print("=== Wyniki ===")
    print(classification_report(MoveType_test, MoveType_pred))