import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model(training_data_file, model_file='Data/model.pkl'):
    with open(training_data_file, 'rb') as f:
        MoveData_train, MoveType_train = pickle.load(f)

    clf = RandomForestClassifier(n_estimators=100, random_state=37)
    clf.fit(MoveData_train, MoveType_train)
    with open(model_file, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Model zapisany do {model_file}")