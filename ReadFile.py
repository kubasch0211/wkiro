import c3d


def readFile(path):
    selected_markers = ['LANK', 'RANK', 'LKNE', 'RKNE', 'LASI', 'RASI', 'LSHO', 'RSHO', 'LELB', 'RELB', 'LWRA', 'RWRA']

    with open(path, 'rb') as handle:
        reader = c3d.Reader(handle)
        first_frame_number = reader.first_frame

        labels = reader.point_labels  # gets labels of the columns
        marker_indices = [i for i, label in enumerate(labels) if label.strip() in selected_markers]  # gets IDs of selected_markers elements

        for frame_number, points, analog in reader.read_frames():  # for each frame
            print(f"Frame {frame_number-first_frame_number}:")
            for index, marker_name in zip(marker_indices, selected_markers):  # for each marker
                print(f"  {marker_name}: {points[index, :3]}")
            print(f"\n\n")
