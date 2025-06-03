import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model(data_file='Data/data.pkl', model_file='Data/model.pkl'):
    with open(data_file, 'rb') as f:
        Coords, MoveType = pickle.load(f)
    Coords_train, Coords_test, MoveType_train, MoveType_test = train_test_split(Coords, MoveType, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(Coords_train, MoveType_train)
    with open(model_file, 'wb') as f:
        pickle.dump((clf, Coords_test, MoveType_test), f)
    print(f"Model zapisany do {model_file}")