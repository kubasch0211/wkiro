import os
import numpy as np
import c3d
import ezc3d
import pickle


def calculate_angle(point_a, point_b, point_c):
    a = np.array(point_a)
    b = np.array(point_b)
    c = np.array(point_c)

    vector_ba = a - b
    vector_bc = c - b
    length_ba = np.linalg.norm(vector_ba)
    length_bc = np.linalg.norm(vector_bc)

    if length_ba < 1e-10 or length_bc < 1e-10:
        return None

    cos_angle = np.dot(vector_ba, vector_bc) / (length_ba * length_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_degrees = np.degrees(np.arccos(cos_angle))
    return angle_degrees


def extract_features_from_file(file_path):
    try:
        # Sprawdź czy plik istnieje
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Plik {file_path} nie istnieje")

        c3d = ezc3d.c3d(file_path)
        points_data = c3d['data']['points']  # dane markerów
        point_labels = c3d['parameters']['POINT']['LABELS']['value']  # nazwy markerów

        # wymiary danych
        n_markers = points_data.shape[1]
        n_frames = points_data.shape[2]

        # słownik markerów
        markers_data = {}
        for i, label in enumerate(point_labels):
            clean_label = label.strip()
            # koordynaty x, y, z
            markers_data[clean_label] = points_data[:3, i, :].T  # Transpozycja: [n_frames, 3]

        required_markers = ['LANK', 'RANK', 'LKNE', 'RKNE', 'LASI', 'RASI', 'LSHO', 'RSHO', 'LELB', 'RELB', 'LWRA', 'RWRA']
    #                      [ 81,     83,     85,     88,     90,     92,     95,     96,     102,    106,    115,    119]
    #                      [ 0,      1,      2,      3,      4,      5,      6,      7,      8,      9,      10,     11]

        missing_markers = [marker for marker in required_markers if marker not in markers_data]
        if missing_markers:
            raise ValueError(f"Brakujące markery: {missing_markers}")

        # kąty do obliczenia
        angle_definitions = [
            ('LANK', 'LKNE', 'LASI'),  # kostka-kolano-biodro L
            ('RANK', 'RKNE', 'RASI'),  # kostka-kolano-biodro R
            ('LKNE', 'LASI', 'LSHO'),  # kolano-biodro-ramię L
            ('RKNE', 'RASI', 'RSHO'),  # kolano-biodro-ramię R
            ('LASI', 'LSHO', 'LELB'),  # biodro-ramię-łokieć L
            ('RASI', 'RSHO', 'RELB'),  # biodro-ramię-łokieć R
            ('LSHO', 'LELB', 'LWRA'),  # ramię-łokieć-nadgarstek L
            ('RSHO', 'RELB', 'RWRA')  # ramię-łokieć-nadgarstek R
        ]
        features = []

        for marker1, marker2, marker3 in angle_definitions:
            angles_over_time = []

            for frame in range(n_frames):
                point_a = markers_data[marker1][frame]
                point_b = markers_data[marker2][frame]  # wierzchołek kąta
                point_c = markers_data[marker3][frame]

                if (not np.any(np.isnan([point_a, point_b, point_c])) and
                        not np.any(np.isinf([point_a, point_b, point_c]))):
                    angle = calculate_angle(point_a, point_b, point_c)
                    if angle is not None:
                        angles_over_time.append(angle)

            if len(angles_over_time) > 0:
                mean_angle = np.mean(angles_over_time)
                # std_angle = np.std(angles_over_time)

                # features.extend([mean_angle, std_angle])
                features.extend([mean_angle])
            else:
                features.extend([0.0, 0.0])

        return features

    except Exception as e:
        print(f"Błąd podczas przetwarzania pliku {file_path}: {e}")
        return []


# def extract_features_from_file(path):
#     selected_markers = ['LANK', 'RANK', 'LKNE', 'RKNE', 'LASI', 'RASI', 'LSHO', 'RSHO', 'LELB', 'RELB', 'LWRA', 'RWRA']
# #                      [ 81,     83,     85,     88,     90,     92,     95,     96,     102,    106,    115,    119]
# #                      [ 0,      1,      2,      3,      4,      5,      6,      7,      8,      9,      10,     11]
#
#     with open(path, 'rb') as handle:
#         reader = c3d.Reader(handle)
#         labels = reader.point_labels
#         marker_indices = [i for i, label in enumerate(labels) if label.strip() in selected_markers]
#         frames = [ [points[i, :3] for i in marker_indices] for _, points, _ in reader.read_frames() ]
#
#     # for i in range(len(frames)):
#     #     print("frame: ", frames[i])
#     #     print("frame1: ", frames[i][0])
#     #     print("coord: ", MoveData[i], '\n')
#
#     for frame in frames:
#         LANK = frame[0]
#         RANK = frame[1]
#         LKNE = frame[2]
#         RKNE = frame[3]
#         LASI = frame[4]
#         RASI = frame[5]
#         LSHO = frame[6]
#         RSHO = frame[7]
#         LELB = frame[8]
#         RELB = frame[9]
#         LWRA = frame[10]
#         RWRA = frame[11]
#
#         # KĄT KOLANA: kostka-kolano-biodro, LANK_LKNE_LASI, RANK_RKNE_RASI
#         # LANK_LKNE = LANK-LKNE
#         # LASI_LKNE = LASI-LKNE
#         # LANK_LKNE_LASI = np.degrees(np.arccos(np.dot(LANK_LKNE, LASI_LKNE) / (np.linalg.norm(LANK_LKNE) * np.linalg.norm(LASI_LKNE))))
#         # RANK_RKNE = RANK-RKNE
#         # RASI_RKNE = RASI-RKNE
#         # RANK_RKNE_RASI = np.degrees(np.arccos(np.dot(RANK_RKNE, RASI_RKNE) / (np.linalg.norm(RANK_RKNE) * np.linalg.norm(RASI_RKNE))))
#         # print(LANK_LKNE_LASI, RANK_RKNE_RASI)
#
#         # KĄT BIODRA: kolano-biodro-ramie, LKNE_LASI_LSHO, RKNE_RASI_RSHO
#         # KĄT RAMIENIA: biodro-ramie-łokieć, LASI_LSHO_LELB, RASI_RSHO_RELB
#         # KĄT ŁOKCIA: ramie-łokieć-nadgarstek, LSHO_LELB_LWRA, RSHO_RELB_RWRA
#
#     frames = np.array(frames)
#     MoveData = frames[:, 0, :]
#
#     velocities = np.linalg.norm(np.diff(MoveData, axis=0), axis=1)
#     return {
#         'mean_velocity': np.mean(velocities),
#         'std_velocity': np.std(velocities),
#         'range_z': np.ptp(MoveData[:, 2])
#     }

def process_dataset(root_dir, output_file='Data/data.pkl'):
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
                                MoveData.append(list(feats))
                                MoveType.append(label)
                                print(f"Processed: {full_path}")
                            except Exception as e:
                                print(f"Error processing {full_path}: {e}")

    # print(MoveData)
    # print(len(MoveData))
    # print(MoveType)
    # print(len(MoveType))

    with open(output_file, 'wb') as f:
        pickle.dump((np.array(MoveData), np.array(MoveType)), f)
    print(f"Dane zapisane do {output_file}")