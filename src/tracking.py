import numpy as np
import json
import os

ROOT_DIR = os.path.abspath("./")

FILTERED_DIR = os.path.join(ROOT_DIR, "filtered")
TRACKED_DIR = os.path.join(ROOT_DIR, "tracked")

TEST_NAME = "test2_det"

DISTANCE_THRESHOLD = 1

MAX_TRACK_DIST = 7
MAX_ERROR_DIST = 50 # speed + error
MAX_SPEED = 6 # pixel/frame
MAX_DOGS = 2
MAX_PERSONS = 2
PREV_FRAME_LENGHT = 80

CAM = 1

def find_previous_object_id(prev_frames, mask_com, object_class):
    mask_com_pt = np.array(mask_com)
    object_id = -1
    score = 0
    scores = {}
    time = 0
    for frame in prev_frames:
        time += 1
        for object in frame:
            if object_class == object['class']:
                dist = np.linalg.norm(np.array(object['mask_com']) - mask_com_pt)
                if dist < MAX_TRACK_DIST:
                    found_object_id = object['id']
                    if found_object_id not in scores:
                        scores[found_object_id] = 0
                    scores[found_object_id] += 1 - time / PREV_FRAME_LENGHT

    for id in scores:
        if scores[id] > score:
            object_id = id
            score = scores[id]

    return object_id, score

def check_scores(frame, object_score, object_class, object_id):
    min_score_idx = -1
    for i, object in enumerate(frame):
        if object['class'] == object_class and object['id'] == object_id and object['score'] < object_score:
            min_score_idx = i

    return min_score_idx

def find_last_seen_entries(prev_frames, object_class, object_id):
    prev_pts = []
    time = 0
    for frame in prev_frames:
        time += 1
        for object in frame:
            if object['class'] == object_class and object['id'] == object_id:
                prev_pts.append([np.array(object['mask_com']), object['score'], time])

    return prev_pts

def find_strongest_pt(pts):
    biggest_score = -1
    best_pt = None
    time = None
    for pt in pts:
        dists = []
        for pt2 in pts:
            dist = np.linalg.norm(pt[0] - pt2[0])
            dists.append(dist)
        avg_dist = sum(dists) / len(dists)
        if avg_dist == 0:
            score = 99999
        else:
            score = 1 / avg_dist * PREV_FRAME_LENGHT / pt[2]

        if score > biggest_score:
            biggest_score = score
            best_pt = pt[0]
            time = pt[2]

    return best_pt, biggest_score, time

def find_nearest_id(mask_com, id_points, object_class):
    mask_com_pt = np.array(mask_com)
    min_dist = 99999999
    min_dist_id_point = None
    for id_point in id_points:
        if id_point[2] is not None and id_point[0] == object_class:
            dist = np.linalg.norm(id_point[2] - mask_com_pt)
            if dist < min_dist:
                min_dist = dist
                min_dist_id_point = id_point

    return min_dist_id_point, min_dist

# Jó irányba halad-e, nem lett használva
def check_id_validity(prev_frames, mask_com, id_dist, found_time, object_class, object_id):
    mask_com_pt = np.array(mask_com)
    is_valid = True
    time = 0
    for frame in prev_frames:
        time += 1
        for object in frame:
            if time < found_time and object['class'] == object_class and object['id'] == object_id:
                dist = np.linalg.norm(np.array(object['mask_com']) - mask_com_pt)
                if id_dist < dist - DISTANCE_THRESHOLD:
                    is_valid = False

    return is_valid

def make_id_dicts():
    dog_ids = []
    person_ids = []
    for j in range(MAX_PERSONS):
        person_ids.append(j+1)
    for i in range(MAX_DOGS):
        dog_ids.append(MAX_PERSONS+i+1)

    return {"person": person_ids, "dog": dog_ids}

def print_frame(frame_id, frame):
    for entry in frame:
        print(frame_id, entry)

def track_comdata(com_data):
    tracked_data = {}
    prev_frames = []
    cam_id = CAM
    # For frame in frames
    for frame_id, frame_objects in com_data.items():
        frame = []
        untracked_frame_objects = []
        id_dict = make_id_dicts()
        # For entry in frame
        for object in frame_objects:
            object_id, score = find_previous_object_id(prev_frames, object['mask_com'], object['class'])

            if object_id == -1 and id_dict[object['class']]:
                untracked_frame_objects.append(object)

            elif object_id != -1:
                if object_id not in id_dict[object['class']]:
                    min_score_idx = check_scores(frame, score, object['class'], object_id)

                    if min_score_idx != -1:
                        del frame[min_score_idx]

                        object_info = {'cam_id': cam_id, 'id': object_id, 'mask_com': object['mask_com'], 'ground_point': object['ground_point'], 'class': object['class'], 'score': score, 'height': object['height'], 'width': object['width']}
                        frame.append(object_info)

                else:
                    id_dict[object['class']].remove(object_id)
                    object_info = {'cam_id': cam_id, 'id': object_id, 'mask_com': object['mask_com'], 'ground_point': object['ground_point'], 'class': object['class'], 'score': score, 'height': object['height'], 'width': object['width']}
                    frame.append(object_info)

        frame_added = True
        while frame_added:
            frame_added = False
            id_points = []
            for object_class, free_id_list in id_dict.items():
                for free_id in free_id_list:
                    last_seen_pts = find_last_seen_entries(prev_frames, object_class, free_id)
                    if last_seen_pts:
                        strongest_pt, score, time = find_strongest_pt(last_seen_pts)
                    else:
                        strongest_pt = None
                        time = None
                        score = None
                    id_points.append([object_class, free_id, strongest_pt, time])

            id_points_bind = []
            for un_object in untracked_frame_objects:
                nearest_id_pt, nearest_dist = find_nearest_id(un_object['mask_com'], id_points, un_object['class'])
                if nearest_id_pt:
                    id_points_bind.append([nearest_id_pt[0], nearest_id_pt[1], nearest_id_pt[2], nearest_id_pt[3], nearest_dist, un_object])

            invalid_binds = []
            for bind in id_points_bind:
                for bind2 in id_points_bind:
                    if bind[0] == bind2[0] and bind[1] == bind2[1] and bind[5] != bind2[5]:
                        if bind[4] > bind2[4]:
                            id_points_bind.remove(bind)
                            invalid_binds.append([bind[0], bind[1], bind[5]])
                        else:
                            id_points_bind.remove(bind2)
                            invalid_binds.append([bind2[0], bind2[1], bind2[5]])

            for bind in id_points_bind:
                if MAX_SPEED * bind[3] + MAX_ERROR_DIST > bind[4]:
                    object = bind[5]
                    object_class = bind[0]
                    object_id = bind[1]
                    score = 1 / bind[4]
                    untracked_frame_objects.remove(bind[5])
                    id_dict[object_class].remove(object_id)
                    object_info = {'cam_id': cam_id, 'id': object_id, 'mask_com': object['mask_com'], 'ground_point': object['ground_point'], 'class': object['class'], 'score': score, 'height': object['height'], 'width': object['width']}
                    frame.append(object_info)
                    frame_added = True
                else:
                    invalid_binds.append([bind[0], bind[1], bind[5]])

        score = 0
        for object_class, free_id_list in id_dict.items():
            for free_id in free_id_list:
                for un_object in untracked_frame_objects:
                    if un_object['class'] == object_class and [object_class, free_id, un_object] not in invalid_binds:
                        object_id = free_id
                        id_dict[un_object['class']].remove(object_id)
                        object_info = {'cam_id': cam_id, 'id': object_id, 'mask_com': un_object['mask_com'], 'ground_point': un_object['ground_point'], 'class': un_object['class'], 'score': score, 'height': un_object['height'], 'width': un_object['width']}
                        frame.append(object_info)
                        break

        prev_frame = frame.copy()
        if len(prev_frames) > PREV_FRAME_LENGHT - 1:
            del prev_frames[-1]
        prev_frames.insert(0, prev_frame)
        print_frame(frame_id, frame)
        tracked_data[frame_id] = frame

    return tracked_data



if __name__ == "__main__":
    with open(os.path.join(FILTERED_DIR,  TEST_NAME + ".json"), "r") as json_file:
        com_data = json.load(json_file)

    tracked_data = track_comdata(com_data)

    with open(os.path.join(TRACKED_DIR, TEST_NAME + ".json"), "w") as json_file:
        json.dump(tracked_data, json_file, ensure_ascii=False, indent=4)
