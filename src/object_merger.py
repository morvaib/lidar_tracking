import os
import json

ROOT_DIR = os.path.abspath("./")

MERGED_DIR = os.path.join(ROOT_DIR, "datastream/yolo7_deepsort/merged")
DOG_PATH = os.path.join(ROOT_DIR, "datastream/yolo7_deepsort/filtered/dogs")
HUMAN_PATH = os.path.join(ROOT_DIR, "datastream/yolo7_deepsort/filtered/humans")

TEST_NAME = "05_23_sync_det"

CAM_EXCLUDE = [6]

#read in files
def read_files(filenames, path):
    com_data = []
    for filename in filenames:
        cam_num = int(filename[3])
        file_path = os.path.join(path, filename)
        with open(os.path.join(file_path), "r") as json_file:
            data = json.load(json_file)
        com_data.append([cam_num, data])

    return com_data

#merge the objects from different cameras
def merge_cameras(tracked_data):
    multi_cam_data = {}
    for cam_data in tracked_data:
        cam_id = cam_data[0]
        if cam_id not in CAM_EXCLUDE:
            for frame_id, frame_data in cam_data[1].items():
                if frame_id not in multi_cam_data.keys():
                    multi_cam_data[frame_id] = []
                for object in frame_data:
                    object_info = {'cam_id': cam_id, 'id': object['id'], 'mask_com': object['mask_com'], \
                                   'ground_point': object['ground_point'], 'class': object['class'], \
                                    'score': object['score'], 'height': object['height'], 'width': object['width']}
                    multi_cam_data[frame_id].append(object_info)
                    #print(frame_id, object_info)

    return multi_cam_data

#merge the different objects
def merge_objects(dogs, humans):
    merged_objetcs = dogs
    for frame_id, frame_data in humans.items():
        for object in frame_data:
            merged_objetcs[frame_id].append(object)
    return merged_objetcs

if __name__ == "__main__":
    tracked_path = os.path.join(MERGED_DIR, TEST_NAME)
    dog_path = os.path.join(DOG_PATH, TEST_NAME)
    human_path = os.path.join(HUMAN_PATH, TEST_NAME)
    _, _, filenames_dogs = next(os.walk(dog_path))
    _, _, filenames_humans = next(os.walk(human_path))
    dog_data = read_files(filenames_dogs, dog_path)
    human_data = read_files(filenames_humans, human_path)

    multi_dog = merge_cameras(dog_data)
    multi_human = merge_cameras(human_data)
    merged_data = merge_objects(multi_dog, multi_human)

    with open(os.path.join(MERGED_DIR, TEST_NAME + ".json"), "w") as json_file:
        json.dump(merged_data, json_file, ensure_ascii=False, indent=4)