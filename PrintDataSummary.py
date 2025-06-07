import os
import c3d
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def get_sequence_lengths(root_dir):
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
                for speed in os.listdir(trial_path):
                    data_path = os.path.join(trial_path, speed, 'Post_Process')
                    if not os.path.exists(data_path):
                        continue
                    for file in os.listdir(data_path):
                        if file.endswith('.c3d'):
                            full_path = os.path.join(data_path, file)
                            try:
                                with open(full_path, 'rb') as handle:
                                    reader = c3d.Reader(handle)
                                    frame_count = sum(1 for _ in reader.read_frames())
                                    lengths.append(frame_count)
                                    print(f"{full_path}: {frame_count} klatek")
                            except Exception as e:
                                print(f"Błąd w {full_path}: {e}")

    return lengths

# Użycie:
root_dir = 'C:\WKIRO\wkiro\TrainDataSet'  # <-- ustaw własną ścieżkę
lengths = get_sequence_lengths(root_dir)

# Statystyki
lengths = np.array(lengths)
print("\n--- Statystyki długości sekwencji ---")
print(f"Min:     {np.min(lengths)}")
print(f"Max:     {np.max(lengths)}")
print(f"Średnia: {np.mean(lengths):.2f}")
print(f"Mediana: {np.median(lengths)}")
print(f"10 percentyl: {np.percentile(lengths, 10)}")
print(f"90 percentyl: {np.percentile(lengths, 90)}")
print(f"Liczba plików: {len(lengths)}")

# (Opcjonalnie) Wykres
plt.hist(lengths, bins=30, color='skyblue', edgecolor='black')
plt.title('Rozkład długości sekwencji (.c3d)')
plt.xlabel('Liczba klatek')
plt.ylabel('Liczba plików')
plt.grid(True)
plt.show()
