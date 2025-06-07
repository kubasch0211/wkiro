import c3d

'''
Odczytywanie i wypisanie dostepnych w pliku .c3d markerow
'''

with open("C:\WKIRO\wkiro\TrainDataSet\CF027\Session1\Overground_Run\Run_Comfortable\Post_Process\Run_Comfortable1.c3d", "rb") as f:
    reader = c3d.Reader(f)
    labels = reader.point_labels
    print("Dostępne markery:", labels)

    for i, (_, points, _) in enumerate(reader.read_frames()):
        print(f"Klatka {i}:")
        for j, label in enumerate(labels):
            pos = points[j, :3]  # X, Y, Z
            print(f"  {label}: {pos}")
        if i == 1:  # tylko dwie klatki
            break