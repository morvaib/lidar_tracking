import numpy as np
import os
import json

ROOT_DIR = os.path.abspath("./")

TRACKED_DIR = os.path.join(ROOT_DIR, "tracked")

TEST_NAME = "test2_det"

# Frame interpolate range
MAX_INTERPOLATE_LENGHT = 40
MIN_INTERPOLATE_LENGHT = 0

MIN_SCORE = 0.5

MAX_DOGS = 2
MAX_PERSONS = 2

def insert_points(tracked_data, curr_frame_id, missing_frames, id_num, category):
    for object in tracked_data[str(curr_frame_id - missing_frames - 1)]:
        if object['id'] == id_num and object['class'] == category:
            com_start_x = object['mask_com'][0]
            com_start_y = object['mask_com'][1]
            floor_start_x = object['ground_point'][0]
            floor_start_y = object['ground_point'][1]
            cam_num = object['cam_id']
            score = object['score']
            height = object['height']
            width = object['width']

    for object in tracked_data[str(curr_frame_id)]:
        if object['id'] == id_num and object['class'] == category:
            com_end_x = object['mask_com'][0]
            com_end_y = object['mask_com'][1]
            floor_end_x = object['ground_point'][0]
            floor_end_y = object['ground_point'][1]

    interpolated_data = tracked_data.copy()
    distance = np.linalg.norm(np.array([com_start_x, com_start_y]) - np.array([com_end_x, com_end_y]))
    if (score > MIN_SCORE or missing_frames < MIN_INTERPOLATE_LENGHT):
        for i in range(missing_frames):
            com_x = com_start_x + (i+1) * (com_end_x - com_start_x) / missing_frames
            com_y = com_start_y + (i+1) * (com_end_y - com_start_y) / missing_frames
            floor_x = floor_start_x + (i+1) * (floor_end_x - floor_start_x) / missing_frames
            floor_y = floor_start_y + (i+1) * (floor_end_y - floor_start_y) / missing_frames
            mask_com = [int(com_x), int(com_y)]
            ground_point = [int(floor_x), int(floor_y)]
            frame_id = str(curr_frame_id - missing_frames + i)
            object_info = {'cam_id': cam_num, 'id': id_num, 'mask_com': mask_com, 'ground_point': ground_point, 'class': category, 'score': 0.1, 'height': height, 'width': width}
            print(frame_id, object_info)
            interpolated_data[frame_id].append(object_info)

    return interpolated_data

def interpolate(tracked_data, id_num, category):
    interpolated_data = tracked_data.copy()
    missing_frames = 0
    is_prev_missing = False
    
    for frame_id, frame_data in tracked_data.items():
        is_missing = True
        for object in frame_data:
            if object['id'] == id_num and object['class'] == category:
                is_missing = False
        
        if is_missing:
            missing_frames += 1

        if is_prev_missing and not is_missing:
            if int(frame_id) > missing_frames + 1 and missing_frames < MAX_INTERPOLATE_LENGHT:
                interpolated_data = insert_points(interpolated_data, int(frame_id), missing_frames, id_num, category)
            missing_frames = 0

        is_prev_missing = is_missing

    return interpolated_data

def interpolate_all(tracked_data):
    # interpolate all ID-s separate
    interpolated_data = tracked_data.copy()
    for i in range(MAX_PERSONS):
        interpolated_data = interpolate(interpolated_data, i+1, 'person')
    for i in range(MAX_DOGS):
        interpolated_data = interpolate(interpolated_data, MAX_PERSONS+i+1, 'dog')
    

    return interpolated_data


if __name__ == "__main__":
    with open(os.path.join(TRACKED_DIR,  TEST_NAME + ".json"), "r") as json_file:
        tracked_data = json.load(json_file)

    interpolated_data = interpolate_all(tracked_data)

    with open(os.path.join(TRACKED_DIR, TEST_NAME + ".json"), "w") as json_file:
        json.dump(interpolated_data, json_file, ensure_ascii=False, indent=4)