import os
import numpy as np
import c3d
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

SELECTED_MARKERS = ['LKneeAngles', 'RKneeAngles', 'LHipAngles', 'RHipAngles',
                    'LShoulderAngles', 'RShoulderAngles', 'LElbowAngles', 'RElbowAngles']

WINDOW_SIZE = 100  # długość okna (liczba klatek)
STEP_SIZE = 50     # przesunięcie okna (np. połowa długości okna)

def extract_windows_from_file(path, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    with open(path, 'rb') as handle:
        reader = c3d.Reader(handle)
        labels = reader.point_labels
        marker_indices = [i for i, label in enumerate(labels) if label.strip() in SELECTED_MARKERS]

        frames = []
        for _, points, _ in reader.read_frames():
            frame = []
            for idx in marker_indices:
                frame.extend(points[idx, :3])  # pobierz x,y,z
            frames.append(frame)

    frames = np.array(frames)  # (num_frames, num_markers*3)

    # dzielenie na okna
    windows = []
    for start in range(0, len(frames) - window_size + 1, step_size):
        window = frames[start:start+window_size]
        windows.append(window)
    return windows

def process_dataset(root_dir, output_file='processed_data.pkl'):
    X, y = [], []
    lengths = []

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

                label = 0 if 'Walk' in trial_type else 1  # 0=walk, 1=run

                for speed in os.listdir(trial_path):
                    data_path = os.path.join(trial_path, speed, 'Post_Process')
                    if not os.path.exists(data_path):
                        continue

                    for file in os.listdir(data_path):
                        if file.endswith('.c3d'):
                            full_path = os.path.join(data_path, file)
                            try:
                                windows = extract_windows_from_file(full_path)
                                X.extend(windows)
                                y.extend([label] * len(windows))
                                lengths.append(len(windows))
                                print(f"Przetworzono {full_path}: {len(windows)} okien")
                            except Exception as e:
                                print(f"Błąd w {full_path}: {e}")

    X = np.array(X)
    y = np.array(y)

    with open(output_file, 'wb') as f:
        pickle.dump({'X': X, 'y': y}, f)
    print(f"Dane zapisane do {output_file}")

    # Podsumowanie
    print("\n--- Statystyki długości okien na plik ---")
    print(f"Min: {np.min(lengths)}")
    print(f"Max: {np.max(lengths)}")
    print(f"Średnia: {np.mean(lengths):.2f}")
    print(f"Mediana: {np.median(lengths)}")
    print(f"Liczba plików: {len(lengths)}")

    # Wykres histogramu długości okien
    plt.hist(lengths, bins=20, alpha=0.7)
    plt.xlabel('Liczba okien z pliku')
    plt.ylabel('Liczba plików')
    plt.title('Rozkład liczby okien na plik')
    plt.show()

    # --- Testy weryfikacyjne ---

    print("\n--- Weryfikacja danych ---")
    print(f"X shape: {X.shape}")  # (liczba_okien, WINDOW_SIZE, liczba_cech)
    print(f"y shape: {y.shape}")  # (liczba_okien,)

    unique_labels = np.unique(y)
    print(f"Unikalne etykiety: {unique_labels}")

    all_correct_length = all(window.shape[0] == WINDOW_SIZE for window in X)
    print(f"Czy wszystkie okna mają długość {WINDOW_SIZE}? {all_correct_length}")

    zero_windows = [i for i, w in enumerate(X) if np.all(w == 0)]
    print(f"Liczba całkowicie zerowych okien: {len(zero_windows)}")

    # Wizualizacja przykładowego okna (pierwszego niezerowego)
    for i, w in enumerate(X):
        if not np.all(w == 0):
            sample_window = w
            break

    plt.figure(figsize=(12, 6))
    plt.plot(sample_window[:, 0], label='LKneeAngles_x')
    plt.plot(sample_window[:, 1], label='LKneeAngles_y')
    plt.plot(sample_window[:, 2], label='LKneeAngles_z')
    plt.title('Przykładowy marker LKneeAngles w czasie (okno)')
    plt.xlabel('Klatka')
    plt.ylabel('Wartość')
    plt.legend()
    plt.show()
