"""
Wyswietlenie zestawienia dotyczacego danych programowych.
"""

import os
import c3d
import numpy as np
import matplotlib
import seaborn as sns
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
root_dir = 'C:\WKIRO\wkiro\TrainDataSet'
lengths = get_sequence_lengths(root_dir)

# Statystyki
lengths = np.array(lengths)
print("\n--- Statystyki liczby klatek ---")
print(f"Min:     {np.min(lengths)}")
print(f"Max:     {np.max(lengths)}")
print(f"Srednia: {np.mean(lengths):.2f}")
print(f"Liczba plikow: {len(lengths)}")

plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Histogram
n, bins, patches = plt.hist(lengths, bins=30, color='skyblue', edgecolor='black', alpha=0.85)

# Tytuł i etykiety
plt.title('Rozklad liczby klatek w plikach', fontsize=14)
plt.xlabel('Liczba klatek', fontsize=12)
plt.ylabel('Liczba plikow', fontsize=12)

# Siatka i styl
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# Dodanie wartości liczbowych nad słupkami
for count, x in zip(n, bins[:-1]):
    if count > 0:
        plt.text(x + (bins[1] - bins[0]) / 2, count + 0.5, str(int(count)),
                 ha='center', va='bottom', fontsize=9, rotation=0)

plt.tight_layout()
plt.show()
