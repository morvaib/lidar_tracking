import os
import json
import copy

import numpy as np

ROOT_DIR = os.path.abspath("./datastream/yolo7_deepsort") #detectron2_tracking

PROJECTED_DIR = os.path.join(ROOT_DIR, "projected")
PROJECTED_POINTS_DIR = os.path.join(PROJECTED_DIR, "points")
PROJECTED_CLUSTER_CENTERS_DIR = os.path.join(PROJECTED_DIR, "cluster_centers")

TEST_NAME = "04_03_17_11_sync_det"
# Labor parameters in meter
LAB_X = 5.4
LAB_Y = 6.27

MAX_DOGS = 2
MAX_PERSONS = 2
CAMS = 5
MAX_CLUSTER_RANGE = 0.5 # restriction
MAX_LAB_PREV_DIST = 0.2 # kmeans init track dist
MAX_LAB_TRACK_DIST = 0.5 # center tracking track dist
MAX_LAB_SPEED = 0.3
MAX_LAB_ERROR_DIST = 1 # speed + error

PREV_LAB_FRAME_LENGHT = 10


def check_border_points(projected_pt):
    is_border = False
    if projected_pt[0] < 0 or projected_pt[0]  > LAB_X \
    or projected_pt[1] < 0 or projected_pt[1] > LAB_Y:
        is_border = True

    return is_border


def make_id_dicts():
    dog_ids = []
    person_ids = []
    for j in range(MAX_PERSONS):
        person_ids.append(j+1)
    for i in range(MAX_DOGS):
        dog_ids.append(MAX_PERSONS+i+1)

    return {"person": person_ids, "dog": dog_ids}


def calc_center(cluster):
    sum_num = 0
    sum_x = 0
    sum_y = 0
    for object in cluster['objects']:
        sum_num += 1
        sum_x += object['projected_point'][0]
        sum_y += object['projected_point'][1]

    return [sum_x/sum_num, sum_y/sum_num]


def check_scores(frame, cluster_score, cluster_class, cluster_id):
    min_score_idx = -1
    for i, cluster in enumerate(frame):
        if cluster['class'] == cluster_class and cluster['id'] == cluster_id and cluster['score'] < cluster_score:
            min_score_idx = i

    return min_score_idx


def find_closest_prev_cluster(prev_frames, cluster, cluster_class):
    cluster_center = np.array(cluster['center'])
    cluster_id = -1
    score = 0
    scores = {}
    time = 0
    for frame in prev_frames:
        time += 1
        for cluster_info in frame:
            if cluster_class == cluster_info['class']:
                dist = np.linalg.norm(np.array(cluster_info['center']) - cluster_center)
                if dist < MAX_LAB_TRACK_DIST:
                    found_cluster_id = cluster_info['id']
                    if found_cluster_id not in scores:
                        scores[found_cluster_id] = 0
                    scores[found_cluster_id] += 1 - time / PREV_LAB_FRAME_LENGHT

    for id in scores:
        if scores[id] > score:
            cluster_id = id
            score = scores[id]

    return cluster_id, score

def find_last_seen_entries(prev_frames, cluster_class, cluster_id):
    prev_pts = []
    time = 0
    for frame in prev_frames:
        time += 1
        for cluster in frame:
            if cluster['class'] == cluster_class and cluster['id'] == cluster_id:
                prev_pts.append([np.array(cluster['center']), cluster['score'], time])

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
            score = 1 / avg_dist * PREV_LAB_FRAME_LENGHT / pt[2]

        if score > biggest_score:
            biggest_score = score
            best_pt = pt[0]
            time = pt[2]

    return best_pt, biggest_score, time


def find_nearest_id(center, id_points, cluster_class):
    center_pt = np.array(center)
    min_dist = 99999999
    min_dist_id_point = None
    for id_point in id_points:
        if id_point[2] is not None and id_point[0] == cluster_class:
            dist = np.linalg.norm(id_point[2] - center_pt)
            if dist < min_dist:
                min_dist = dist
                min_dist_id_point = id_point

    return min_dist_id_point, min_dist


def find_last_cluster_coords(cluster_info, prev_frames):
    for prev_frame in prev_frames:
        for prev_cluster_info in prev_frame:
            if prev_cluster_info['id'] == cluster_info['id']:
                return prev_cluster_info['center']


def multi_tracking(data):
    tracked_data = {}
    center_data = {}
    prev_frames = []
    prev_cluster_map = {}
    for frame_id, frame_data in data.items():
        frame_objects = []
        frame = []
        untracked_frame_clusters = []
        id_dict = make_id_dicts()

        start_clusters = {'person':{}, 'dog':{}}
        center_points = {'person':{}, 'dog':{}}

        #Kmeans init
        for cluster_class, id_list in id_dict.items():
            if prev_cluster_map:
                for prev_cluster_id, prev_cluster_info in prev_cluster_map[cluster_class].items():
                    for object in frame_data:
                        if object['class'] == cluster_class and [object['cam_id'], object['id']] in prev_cluster_info['ids']:
                            is_border = check_border_points(object['projected_point'])
                            if not is_border:
                                dist = np.linalg.norm(np.array(object['projected_point']) - np.array(prev_cluster_info['center']))
                                if dist < MAX_LAB_PREV_DIST:
                                    if prev_cluster_id not in start_clusters[object['class']].keys():
                                        start_clusters[object['class']][prev_cluster_id] = {'objects': [object], 'center': object['projected_point'], 'cams': [object['cam_id']], 'class': object['class']}
                                    else:
                                        start_clusters[object['class']][prev_cluster_id]['objects'] += [object]
                                        start_clusters[object['class']][prev_cluster_id]['cams'] += [object['cam_id']]
                                        start_clusters[object['class']][prev_cluster_id]['center'] = calc_center(start_clusters[object['class']][prev_cluster_id])

                                    frame_data.remove(object)

                if len(id_list) > len(start_clusters[cluster_class]):
                    next_id = MAX_PERSONS + MAX_DOGS + 1
                    choosen_cam_ids = []
                    for i in range(len(id_list) - len(start_clusters[cluster_class])):
                        for object in frame_data:
                            is_border = check_border_points(object['projected_point'])
                            if not is_border and object['class'] == cluster_class and [object['id'], object['cam_id']] not in choosen_cam_ids:
                                start_clusters[cluster_class][next_id+i] = {'objects': [], 'center': object['projected_point'], 'cams': [], 'class': cluster_class}
                                choosen_cam_ids.append([object['id'], object['cam_id']])
                                break

            else:
                cams = {1:0,2:0,3:0,4:0,5:0}
                for object in frame_data:
                    if object['class'] == cluster_class:
                        cams[object['cam_id']] += 1

                max_cam = 1
                max_cam_count = 0
                for cam_num, count in cams.items():
                    if count > max_cam_count:
                        max_cam_count = count
                        max_cam = cam_num

                if max_cam_count > len(id_list):
                    max_cam_count = len(id_list)

                choosen_ids = []
                for i in range(max_cam_count):
                    for object in frame_data:
                        is_border = check_border_points(object['projected_point'])
                        if not is_border and object['class'] == cluster_class and object['cam_id'] == max_cam and object['id'] not in choosen_ids:
                            start_clusters[cluster_class][i] = {'objects': [], 'center': object['projected_point'], 'cams': [], 'class': cluster_class}
                            choosen_ids.append(object['id'])
                            break

                choosen_cam_ids = []
                for j in range(len(id_list) - max_cam_count):
                    for object in frame_data:
                        is_border = check_border_points(object['projected_point'])
                        if not is_border and object['class'] == cluster_class and object['cam_id'] != max_cam and [object['id'], object['cam_id']] not in choosen_cam_ids:
                            start_clusters[cluster_class][max_cam_count+j] = {'objects': [], 'center': object['projected_point'], 'cams': [], 'class': cluster_class}
                            choosen_cam_ids.append([object['id'], object['cam_id']])
                            break

        clusters = copy.deepcopy(start_clusters)

        #Kmeans
        prev_clusters = {}
        N = 0
        while prev_clusters != clusters and N < 25:
            N += 1
            prev_clusters = copy.deepcopy(clusters)
            for cluster_class in clusters.keys():
                for cluster_idx, cluster_info in clusters[cluster_class].items():
                    cluster_info['objects'] = copy.deepcopy(start_clusters[cluster_class][cluster_idx]['objects'])
                    cluster_info['cams'] = copy.deepcopy(start_clusters[cluster_class][cluster_idx]['cams'])
                    print(cluster_class, cluster_idx)
            for object in frame_data:
                print(object)
                min_dist = 99999
                min_dist_cluster_idx = None
                for cluster_idx, cluster_info in clusters[object['class']].items():
                    dist = np.linalg.norm(np.array(object['projected_point']) - np.array(cluster_info['center']))
                    if dist < min_dist and object['cam_id'] not in cluster_info['cams']:
                        min_dist = dist
                        min_dist_cluster_idx = cluster_idx

                clusters[object['class']][min_dist_cluster_idx]['objects'].append(object)
                clusters[object['class']][min_dist_cluster_idx]['cams'].append(object['cam_id'])

            for cluster_class in clusters.keys():
                for cluster_idx, cluster_info in clusters[cluster_class].items():
                    if cluster_info['objects']:
                        cluster_info['center'] = calc_center(cluster_info)
                    elif N > 20:
                        del clusters[cluster_class][cluster_idx]


        #center tracking
        for cluster_class in clusters.keys():
            for cluster in clusters[cluster_class].values():
                closest_cluster_id, score = find_closest_prev_cluster(prev_frames, cluster, cluster_class)
                if closest_cluster_id == -1 and id_dict[cluster_class]:
                    untracked_frame_clusters.append(cluster)

                elif closest_cluster_id != -1:
                    if closest_cluster_id not in id_dict[cluster_class]:
                        min_score_idx = check_scores(frame, score, cluster_class, closest_cluster_id)

                        if min_score_idx != -1:
                            del frame[min_score_idx]

                            cluster_info = {'id': closest_cluster_id, 'center': cluster['center'], 'class': cluster_class, 'score': score, 'objects': cluster['objects']}
                            frame.append(cluster_info)

                    else:
                        id_dict[cluster_class].remove(closest_cluster_id)
                        cluster_info = {'id': closest_cluster_id, 'center': cluster['center'], 'class': cluster_class, 'score': score, 'objects': cluster['objects']}
                        frame.append(cluster_info)

        frame_added = True
        while frame_added:
            frame_added = False
            id_points = []
            for cluster_class, free_id_list in id_dict.items():
                for free_id in free_id_list:
                    last_seen_pts = find_last_seen_entries(prev_frames, cluster_class, free_id)
                    if last_seen_pts:
                        strongest_pt, score, time = find_strongest_pt(last_seen_pts)
                    else:
                        strongest_pt = None
                        time = None
                        score = None
                    id_points.append([cluster_class, free_id, strongest_pt, time])

            id_points_bind = []
            for un_cluster in untracked_frame_clusters:
                nearest_id_pt, nearest_dist = find_nearest_id(un_cluster['center'], id_points, un_cluster['class'])
                if nearest_id_pt:
                    id_points_bind.append([nearest_id_pt[0], nearest_id_pt[1], nearest_id_pt[2], nearest_id_pt[3], nearest_dist, un_cluster])

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
                if MAX_LAB_SPEED *  bind[3] + MAX_LAB_ERROR_DIST > bind[4]:
                    cluster = bind[5]
                    cluster_class = bind[0]
                    cluster_id = bind[1]
                    score = 1 / bind[4]
                    untracked_frame_clusters.remove(bind[5])
                    id_dict[cluster_class].remove(cluster_id)
                    cluster_info = {'id': cluster_id, 'center': cluster['center'], 'class': cluster_class, 'score': score, 'objects': cluster['objects']}
                    frame.append(cluster_info)
                    frame_added = True
                else:
                    invalid_binds.append([bind[0], bind[1], bind[5]])

        # delete the smallest cluster if there are too many
        for cluster_class, id_list in id_dict.items():
            while len(id_list) < len(clusters[cluster_class]):
                smallest_cluster_number = 9999
                smallest_cluster_id = None
                for cluster_id, cluster in clusters[cluster_class].items():
                    if len(cluster['objects']) < smallest_cluster_number:
                        smallest_cluster_number = len(cluster['objects'])
                        smallest_cluster_id = cluster_id

                del clusters[cluster_class][smallest_cluster_id]

        score = 0
        for cluster_class, free_id_list in id_dict.items():
            for free_id in free_id_list:
                for un_cluster in untracked_frame_clusters:
                    if un_cluster['class'] == cluster_class and [cluster_class, free_id, un_cluster] not in invalid_binds:
                        cluster_id = free_id
                        id_dict[un_cluster['class']].remove(cluster_id)
                        cluster_info = {'id': cluster_id, 'center': cluster['center'], 'class': cluster_class, 'score': score, 'objects': un_cluster['objects']}
                        frame.append(cluster_info)
                        break

        # Max cluster range check
        for cluster_info in frame:
            to_change = False
            for cluster_object in cluster_info['objects']:
                dist = np.linalg.norm(np.array(cluster_object['projected_point']) - np.array(cluster_info['center']))
                if dist > MAX_CLUSTER_RANGE:
                    to_change = True
                    break

            if to_change:
                new_cluster_info = copy.deepcopy(cluster_info)
                new_cluster_info['objects'] = []
                last_coords = find_last_cluster_coords(cluster_info, prev_frames)
                if last_coords is not None:
                    for cluster_object in cluster_info['objects']:
                        new_dist = np.linalg.norm(np.array(cluster_object['projected_point']) - np.array(last_coords))
                        if new_dist < MAX_CLUSTER_RANGE:
                            new_cluster_info['objects'].append(cluster_object)

                    if new_cluster_info['objects']:
                        new_cluster_info['center'] = calc_center(new_cluster_info)
                        frame.remove(cluster_info)
                        frame.append(new_cluster_info)


        # Save clusters for next iteration
        cluster_map = {'person': {}, 'dog': {}}
        for cluster_info in frame:
            center_points[cluster_info['class']][cluster_info['id']] = cluster_info['center']

            if cluster_info['id'] not in cluster_map[cluster_info['class']].keys():
                cluster_map[cluster_info['class']][cluster_info['id']] = {'ids':[], 'center': cluster_info['center']}

            for object in cluster_info['objects']:
                cluster_map[cluster_info['class']][cluster_info['id']]['ids'].append([object['cam_id'], object['id']])
                object_info = {'cam_id': object['cam_id'], 'id': cluster_info['id'], 'projected_point': object['projected_point'], 'mask_com': object['mask_com'], 'ground_point': object['ground_point'], 'class': object['class'], 'score': object['score'], 'height': object['height'], 'width': object['width']}
                frame_objects.append(object_info)

        prev_cluster_map = cluster_map
        prev_frame = frame.copy()
        if len(prev_frames) > PREV_LAB_FRAME_LENGHT - 1:
            del prev_frames[-1]
        prev_frames.insert(0, prev_frame)

        center_data[frame_id] = center_points
        tracked_data[frame_id] = frame_objects
        print_frame(frame_id, frame_objects)

    return tracked_data, center_data


def print_frame(frame_id, frame_data):
    for object in frame_data:
        print(frame_id, object)


if __name__ == "__main__":
    with open(os.path.join(PROJECTED_POINTS_DIR, TEST_NAME + ".json"), "r") as json_file:
        projected_data = json.load(json_file)

    final_data, clusters = multi_tracking(projected_data)

    with open(os.path.join(PROJECTED_POINTS_DIR, TEST_NAME + ".json"), "w") as json_file:
        json.dump(final_data, json_file, ensure_ascii=False, indent=4)

    with open(os.path.join(PROJECTED_CLUSTER_CENTERS_DIR, TEST_NAME + ".json"), "w") as json_file:
        json.dump(clusters, json_file, ensure_ascii=False, indent=4)
