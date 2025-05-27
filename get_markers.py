import ezc3d

'''
Odczytywanie i wypisanie dostepnych w pliku .c3d markerow
'''

# Ścieżka do pliku .c3d
file_path = "C:/WKIRO/wkiro/DataSet/AJ026/Session1/Overground_Run/Run_Comfortable/Post_Process/Run_Comfortable1.c3d"

# Wczytaj dane
c3d = ezc3d.c3d(file_path)

# Pobierz nazwy markerów (POINT.LABELS)
marker_labels = c3d['parameters']['POINT']['LABELS']['value']

# Wypisz
print("Dostępne markery 3D:")
for label in marker_labels:
    print(label)