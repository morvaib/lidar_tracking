import json
import pickle
import os

ROOT_DIR = os.path.abspath("./")
INPUT_PATH = os.path.join(ROOT_DIR, "outputs")
#INPUT_PATH = os.path.join(INPUT_PATH, "03_23_15_16_39_sync")

def unpack_com_data(entry):
    frame_id = entry[0]
    commask = entry[1].astype(int).tolist()
    combox = entry[2].astype(int).tolist()
    ground_mask = entry[3].astype(int).tolist()
    object_name = entry[4]
    score = float(entry[5])
    height = int(entry[6])
    width = int(entry[7])

    return frame_id, commask, combox, ground_mask, object_name, height, width, score

filenames = next(os.walk(INPUT_PATH))

for filename in filenames[2]:
    if '.p' in filename:
        output_data = pickle.load(open(os.path.join(filenames[0], filename), "rb"))
        print(filename)
        info = {}
        i = 0
        for frame_data in output_data:
            i += 1
            info[i] = []
            for entry in frame_data:
                frame_id, commask, combox, ground_mask, object_name, height, width, score = unpack_com_data(entry)
                entry_info = {'mask_com':commask, 'box_center':combox, 'ground_point': ground_mask, 'class': object_name, 'score': score, 'height': height, 'width':width}
                info[frame_id].append(entry_info)
                if frame_id != i:
                    print('idx error')

        with open(os.path.join(filenames[0], filename.split('.')[0] + '.json'), 'w') as f:
            json.dump(info, f, ensure_ascii=False, indent=4)