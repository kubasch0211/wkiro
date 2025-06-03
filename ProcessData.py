import os
import numpy as np
import c3d
import pickle

def extract_features_from_file(path):
    selected_markers = ['LANK', 'RANK', 'LKNE', 'RKNE', 'LASI', 'RASI', 'LSHO', 'RSHO', 'LELB', 'RELB', 'LWRA', 'RWRA']
    with open(path, 'rb') as handle:
        reader = c3d.Reader(handle)
        labels = reader.point_labels
        marker_indices = [i for i, label in enumerate(labels) if label.strip() in selected_markers]
        frames = [ [points[i, :3] for i in marker_indices] for _, points, _ in reader.read_frames() ]

    frames = np.array(frames)
    coords = frames[:, 0, :]

    # for i in range(len(frames)):
    #     print("frame: ", frames[i])
    #     print("coord: ", coords[i], '\n')

    velocities = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    return {
        'mean_velocity': np.mean(velocities),
        'std_velocity': np.std(velocities),
        'range_z': np.ptp(coords[:, 2])
    }

def process_dataset(root_dir, output_file='Data/data.pkl'):
    Coords, MoveType = [], []
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
                label = 'walk' if 'Walk' in trial_type else 'run'
                for speed in os.listdir(trial_path):
                    data_path = os.path.join(trial_path, speed, 'Post_Process')
                    if not os.path.exists(data_path):
                        continue
                    for file in os.listdir(data_path):
                        if file.endswith('.c3d'):
                            full_path = os.path.join(data_path, file)
                            try:
                                feats = extract_features_from_file(full_path)
                                Coords.append(list(feats.values()))
                                MoveType.append(label)
                                print(f"Processed: {full_path}")
                            except Exception as e:
                                print(f"Error processing {full_path}: {e}")

    # print(Coords)
    # print(len(Coords))
    # print(MoveType)
    # print(len(MoveType))

    with open(output_file, 'wb') as f:
        pickle.dump((np.array(Coords), np.array(MoveType)), f)
    print(f"Dane zapisane do {output_file}")