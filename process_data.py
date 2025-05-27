import os
import numpy as np
import c3d
import pickle
from sklearn.model_selection import train_test_split

SELECTED_MARKERS = ['LKneeAngles', 'RKneeAngles']
MAX_FRAMES = 300  # ustalona długość sekwencji
FEATURES_PER_MARKER = 3 * len(SELECTED_MARKERS)

def extract_sequence(path):
    with open(path, 'rb') as handle:
        reader = c3d.Reader(handle)
        labels = reader.point_labels
        marker_indices = [i for i, label in enumerate(labels) if label.strip() in SELECTED_MARKERS]

        sequence = []
        for _, points, _ in reader.read_frames():
            frame = []
            for idx in marker_indices:
                frame.extend(points[idx, :3])
            sequence.append(frame)

    # pad or cut to MAX_FRAMES
    sequence = np.array(sequence)
    if sequence.shape[0] >= MAX_FRAMES:
        sequence = sequence[:MAX_FRAMES]
    else:
        padding = np.zeros((MAX_FRAMES - sequence.shape[0], FEATURES_PER_MARKER))
        sequence = np.vstack((sequence, padding))

    return sequence

def process_dataset(root_dir, output_file='data.pkl'):
    X, y = [], []
    for subject in os.listdir(root_dir):
        subj_path = os.path.join(root_dir, subject)
        if not os.path.isdir(subj_path):
            continue
        for session in os.listdir(subj_path):
            ses_path = os.path.join(subj_path, session)
            for trial_type in ['Overground_Walk', 'Overground_Run']:
                trial_path = os.path.join(ses_path, trial_type)
                if not os.path.exists(trial_path):
                    continue
                label = 0 if 'Walk' in trial_type else 1
                for speed in os.listdir(trial_path):
                    data_path = os.path.join(trial_path, speed, 'Post_Process')
                    if not os.path.exists(data_path):
                        continue
                    for file in os.listdir(data_path):
                        if file.endswith('.c3d'):
                            full_path = os.path.join(data_path, file)
                            try:
                                seq = extract_sequence(full_path)
                                X.append(seq)
                                y.append(label)
                                print(f"OK: {full_path}")
                            except Exception as e:
                                print(f"Błąd w {full_path}: {e}")
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with open(output_file, 'wb') as f:
        pickle.dump({
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }, f)

    print(f"Zapisano dane do {output_file}")
