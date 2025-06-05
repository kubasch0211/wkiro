import os
import numpy as np
import c3d
import ezc3d
import pickle

def extract_features_from_file(path):
    selected_markers = ['LKneeAngles', 'RKneeAngles', 'LHipAngles', 'RHipAngles', 'LShoulderAngles', 'RShoulderAngles', "LElbowAngles", "RElbowAngles"]

    with open(path, 'rb') as handle:
        reader = c3d.Reader(handle)
        labels = reader.point_labels
        marker_indices = [i for i, label in enumerate(labels) if label.strip() in selected_markers]
        frames = [ [points[i, :3] for i in marker_indices] for _, points, _ in reader.read_frames() ]

    frames = np.array(frames)
    LKneeAngles = [frame[0][0] for frame in frames]
    RKneeAngles = [frame[1][0] for frame in frames]
    LHipAngles = [frame[2][0] for frame in frames]
    RHipAngles = [frame[3][0] for frame in frames]
    LShoulderAngles = [frame[4][0] for frame in frames]
    RShoulderAngles = [frame[5][0] for frame in frames]
    LElbowAngles = [frame[6][0] for frame in frames]
    RElbowAngles = [frame[7][0] for frame in frames]

    return {
        'LKneeAngles': (LKneeAngles),
        'RKneeAngles': (RKneeAngles),
        'LHipAngles': (LHipAngles),
        'RHipAngles': (RHipAngles),
        'LShoulderAngles': (LShoulderAngles),
        'RShoulderAngles': (RShoulderAngles),
        'LElbowAngles': (LElbowAngles),
        'RElbowAngles': (RElbowAngles),
    }

def process_dataset(root_dir, output_file):
    MoveData, MoveType = [], []
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
                                MoveData.append(list(feats.values()))
                                MoveType.append(label)
                                print(f"Processed: {full_path}")
                            except Exception as e:
                                print(f"Error processing {full_path}: {e}")

    with open(output_file, 'wb') as f:
        pickle.dump((np.array(MoveData), np.array(MoveType)), f)
    print(f"Dane zapisane do {output_file}")