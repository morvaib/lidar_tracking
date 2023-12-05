import json
import os

ROOT_DIR = os.path.abspath("./")

TRACKED_DIR = os.path.join(ROOT_DIR, "datastream/detectron2_deepsort/filtered")

TEST_NAME = "04_03_17_22_sync_det"

CAM_EXCLUDE = [5]

def make_one_file(tracked_data):
    multi_cam_data = {}
    for cam_data in tracked_data:
        cam_id = cam_data[0]
        if cam_id not in CAM_EXCLUDE:
            for frame_id, frame_data in cam_data[1].items():
                if frame_id not in multi_cam_data.keys():
                    multi_cam_data[frame_id] = []
                for object in frame_data:
                    object_info = {'cam_id': cam_id, 'id': object['id'], 'mask_com': object['mask_com'], 'ground_point': object['ground_point'], 'class': object['class'], 'score': object['score'], 'height': object['height'], 'width': object['width']}
                    multi_cam_data[frame_id].append(object_info)
                    print(frame_id, object_info)

    return multi_cam_data


def read_files(filenames, path):
    com_data = []
    for filename in filenames:
        cam_num = int(filename[3])
        file_path = os.path.join(path, filename)
        with open(os.path.join(file_path), "r") as json_file:
            data = json.load(json_file)
        com_data.append([cam_num, data])

    return com_data


if __name__ == "__main__":
    tracked_path = os.path.join(TRACKED_DIR, TEST_NAME)
    _, _, filenames = next(os.walk(tracked_path))
    tracked_data = read_files(filenames, tracked_path)
    print(filenames)

    #multi_data = make_one_file(tracked_data)

    # with open(os.path.join(TRACKED_DIR, TEST_NAME + ".json"), "w") as json_file:
    #     json.dump(multi_data, json_file, ensure_ascii=False, indent=4)