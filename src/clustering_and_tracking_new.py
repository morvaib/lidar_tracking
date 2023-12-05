import os
import json
import copy
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
import tqdm

ROOT_DIR = os.path.abspath("./datastream/yolo7_deepsort") #detectron2_tracking
PROJECTED_DIR = os.path.join(ROOT_DIR, "projected")
PROJECTED_POINTS_DIR = os.path.join(PROJECTED_DIR, "points")
PROJECTED_CLUSTER_CENTERS_DIR = os.path.join(PROJECTED_DIR, "cluster_centers")
LASER_DIR ="./laser_coords"
TEST_NAME = "05_23_sync"
# Labor parameters in meter
LAB_X = 5.4
LAB_Y = 6.27

MAX_DOGS = 2
MAX_PERSONS = 2
CAMS = 5
CAM_EXCLUDE = 0
MAX_CLUSTER_RANGE = 0.5 # restriction
MAX_LAB_PREV_DIST = 0.2 # kmeans init track dist
MAX_LAB_TRACK_DIST = 0.35 # center tracking track dist
MAX_LAB_SPEED = 0.3
MAX_LAB_ERROR_DIST = 1 # speed + error

PREV_LAB_FRAME_LENGHT = 10

def make_id_dicts():
    dog_ids = []
    person_ids = []
    for j in range(MAX_PERSONS):
        person_ids.append(j+1)
    for i in range(MAX_DOGS):
        dog_ids.append(i+1)

    return {"person": person_ids, "dog": dog_ids}

def calc_mahalanobis(points, idx):
    if len(points) < 2:
        return 0
    sum_x = 0
    sum_y = 0
    for i in range(len(points)):
        if i != idx:
            sum_x += points[i][0]
            sum_y += points[i][1]
    return np.linalg.norm(np.array([sum_x/(len(points)-1), sum_y/(len(points)-1)]) - np.array(points[idx]))

def calc_center(points):
    if len(points) == 0:
        return [-1, -1]
    sum_x = 0
    sum_y = 0
    for i in range(len(points)):
        sum_x += points[i][0]
        sum_y += points[i][1]
    return  [sum_x/len(points), sum_y/len(points)]

def clustering(data, laser_data):
    clusters = {}
    cluster_centers = {}
    velocity_model = {"person": {}, "dog": {}}
    pbar = tqdm.tqdm(total=len(data))

    id_dicts = make_id_dicts()
    for object_name in id_dicts.keys():
        for id in id_dicts[object_name]:
            velocity_model[object_name][id] = {"dist": 0, "velocity": 0, "pred_point": [0, 0]}

    for frame_num, frame_data in data.items():
        cluster_centers[frame_num] = {"person": {}, "dog": {}}
        pbar.update(1)

        #Kmeans initialization
        for cluster_class in id_dicts.keys():
            if len(id_dicts[cluster_class]) == 0:
                continue
            cams = {1:0, 2:0, 3:0, 4:0, 5:0}
            for object in frame_data:
                if object['class'] == cluster_class:
                        cams[object['cam_id']] += 1

            max_cam_num = 1
            max_cam_count = 0
            for cam_num, count in cams.items():
                if count > max_cam_count:
                    max_cam_count = count
                    max_cam_num = cam_num

            proj_points = []
            for i in range(1, max_cam_count+1):
                for object in frame_data:
                    if object['class'] == cluster_class and object['cam_id'] == max_cam_num and object['projected_point'] not in proj_points and object["cam_id"] != CAM_EXCLUDE:
                        cluster_centers[frame_num][cluster_class][i] = object['projected_point']
                        proj_points.append(object['projected_point'])
                        break

            if cluster_class == "person" and max_cam_count < MAX_PERSONS:
                for j in range(MAX_PERSONS - max_cam_count, MAX_PERSONS + 1):
                    for object in frame_data:
                        if object['class'] == cluster_class and object['projected_point'] not in proj_points and object["cam_id"] != CAM_EXCLUDE:
                            cluster_centers[frame_num][cluster_class][j] = object['projected_point']
                            proj_points.append(object['projected_point'])
                            break

            if cluster_class == "dog" and max_cam_count < MAX_DOGS:
                for k in range(MAX_DOGS - max_cam_count, MAX_DOGS + 1):
                    for object in frame_data:
                        if object['class'] == cluster_class and object['projected_point'] not in proj_points and object["cam_id"] != CAM_EXCLUDE:
                            cluster_centers[frame_num][cluster_class][k] = object['projected_point']
                            proj_points.append(object['projected_point'])
                            break

        clusters[frame_num] = {"person": {}, "dog": {}}
        for obj_type in cluster_centers[frame_num].keys():
            for id in cluster_centers[frame_num][obj_type].keys():
                clusters[frame_num][obj_type][id] = {"center": cluster_centers[frame_num][obj_type][id], "points": []}

        #Kmeans
        n = 0
        while n < 10:
            n += 1
            for object_name in clusters[frame_num].keys():
                for object_id in clusters[frame_num][object_name].keys():
                    if int(frame_num) != 2:
                        clusters[frame_num][object_name][object_id]["points"] = []

            #find nearest center
            for object in frame_data:
                min_dist = 100000
                min_dist_id = None
                for id in clusters[frame_num][object["class"]].keys():
                    dist = np.linalg.norm(np.array(clusters[frame_num][object["class"]][id]["center"])-np.array(object["projected_point"]))
                    if n == 1:
                        clusters[frame_num][object["class"]][id]["points"].append(clusters[frame_num][object["class"]][id]["center"])
                    if dist < min_dist:
                        min_dist = dist
                        min_dist_point = object["projected_point"]
                        min_dist_id = id

                if min_dist_id is not None:
                    clusters[frame_num][object["class"]][min_dist_id]["points"].append(min_dist_point)

            #calc new centers
            for object_name in clusters[frame_num].keys():
                for object_id in clusters[frame_num][object_name].keys():
                    clusters[frame_num][object_name][object_id]["center"] = calc_center(clusters[frame_num][object_name][object_id]["points"])

        #remove too far points
        for object_name in clusters[frame_num].keys():
            for object_id in clusters[frame_num][object_name].keys():
                points_to_remove = []
                for point_idx in range(len(clusters[frame_num][object_name][object_id]["points"])):
                    if calc_mahalanobis(clusters[frame_num][object_name][object_id]["points"], point_idx) > 0.85:
                        points_to_remove.append(clusters[frame_num][object_name][object_id]["points"][point_idx])
                # if 225 < int(frame_num) and int(frame_num) < 300:
                #     print(f"frame{frame_num}")
                #     print(object_name, object_id)
                #     print(clusters[frame_num][object_name][object_id]["points"])
                #     print(clusters[frame_num][object_name][object_id]["center"])
                #     print(points_to_remove)

                for to_remove in points_to_remove:
                    clusters[frame_num][object_name][object_id]["points"].remove(to_remove)

                if len(points_to_remove) > 0:
                    clusters[frame_num][object_name][object_id]["center"] = calc_center(clusters[frame_num][object_name][object_id]["points"])


        #tracking
        if int(frame_num) == 4:
            for object_name in clusters[frame_num].keys():
                act_centers = []
                for object_id in clusters[frame_num][object_name].keys():
                    act_centers.append(clusters[frame_num][object_name][object_id]["center"])

                cost_matrix = np.empty(shape=(len(act_centers), len(act_centers)))
                row_num = 0
                for object_id in clusters[str(int(frame_num) - 1)][object_name].keys():
                    row = []
                    for act_center_num in range(len(act_centers)):
                        row.append(np.linalg.norm(np.array(clusters[str(int(frame_num) - 1)][object_name][object_id]["center"]) - np.array(act_centers[act_center_num])))

                    cost_matrix[row_num] = row
                    row_num += 1
                #Hungarian algorithm
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
                assignment = [(row, col) for row, col in zip(row_indices, col_indices)]
                for row, col in assignment:
                    row = row.item()
                    clusters[frame_num][object_name][row + 1]["center"] = act_centers[col]

        if int(frame_num) > 4:
            #constant velocity model
            prev2 = clusters[str(int(frame_num) - 2)]
            prev1 = clusters[str(int(frame_num) - 1)]
            act = clusters[frame_num]
            predictions = {"person": {}, "dog": {}}
            for object_name in prev2.keys():
                for object_id2 in prev2[object_name].keys():
                    predictions[object_name][object_id2] = {"prev2_center": prev2[object_name][object_id2]["center"], "prev1_center": [], "actual_pred": []}
                    predictions[object_name][object_id2]["prev1_center"] = prev1[object_name][object_id2]["center"]
                    predictions[object_name][object_id2]["actual_pred"] = np.array(prev1[object_name][object_id2]["center"]) + (np.array(prev1[object_name][object_id2]["center"]) - np.array(predictions[object_name][object_id2]["prev2_center"]))

                act_centers = []
                for object_id in act[object_name].keys():
                    act_centers.append(act[object_name][object_id]["center"])

                if len(act_centers) < len(predictions[object_name]):
                    act_centers = []
                    for object_id in predictions[object_name].keys():
                        act_centers.append(predictions[object_name][object_id]["prev1_center"])
                        act[object_name][object_id] = {"center": predictions[object_name][object_id]["prev1_center"], "points": []}

                #Hungarian algorithm
                #             act1             act2             act3
                #cent1  cent1_pred-act1  cent1_pred-act1  cent1_pred-act1
                #cent2  cent2_pred-act1  cent2_pred-act1  cent2_pred-act1
                #cent3  cent3_pred-act1  cent3_pred-act1  cent3_pred-act1
                cost_matrix = np.empty(shape=(len(act_centers), len(act_centers)))

                if 1 < int(frame_num) and int(frame_num) < 306:
                    print(object_name)
                    print(f"act_centers: {act_centers}")
                    print(f"act: {act}")
                    print(f"predictions: {predictions}")
                row_num = 0
                for object_id in predictions[object_name].keys():
                    row = []
                    for act_center_num in range(len(act_centers)):
                        row.append(np.linalg.norm(np.array(predictions[object_name][object_id]["actual_pred"]) - np.array(act_centers[act_center_num])))

                    cost_matrix[row_num] = row
                    row_num +=1

                row_indices, col_indices = linear_sum_assignment(cost_matrix)
                assignment = [(row, col) for row, col in zip(row_indices, col_indices)]
                for row, col in assignment:
                    row = row.item()
                    act[object_name][row + 1]["center"] = act_centers[col]

            clusters[frame_num] = copy.deepcopy(act)

            #check valid id
            to_remove = {"dog": [], "person": []}
            for object_name in clusters[frame_num].keys():
                for id in clusters[frame_num][object_name].keys():
                    if object_name == "dog" and id > MAX_DOGS:
                        to_remove["dog"].append(id)
                    elif object_name == "person" and id > MAX_PERSONS:
                        to_remove["person"].append(id)

            for object_name in to_remove.keys():
                for ob_id in to_remove[object_name]:
                    del clusters[frame_num][object_name][ob_id]
                    del cluster_centers[frame_num][object_name][ob_id]

            #mean of track and center
            for object_name in clusters[frame_num].keys():
                for ob_id in clusters[frame_num][object_name].keys():
                    clusters[frame_num][object_name][object_id]["center"] = [(clusters[frame_num][object_name][object_id]["center"][0] + predictions[object_name][object_id]["actual_pred"].tolist()[0]) / 2,\
                         (clusters[frame_num][object_name][object_id]["center"][1] + predictions[object_name][object_id]["actual_pred"].tolist()[1]) / 2 ]

        for object_name in clusters[frame_num].keys():
            for ob_id in clusters[frame_num][object_name].keys():
                cluster_centers[frame_num][object_name][ob_id] = copy.deepcopy(clusters[frame_num][object_name][ob_id]["center"])

        plot = True
        if plot:
            scale = 1000
            plt.cla()
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.xlim(0, LAB_X*scale)
            plt.ylim(0, LAB_Y*scale)
            plt.text(5000, 5800, frame_num, fontsize=12)
            plt.gca().set_aspect('equal', adjustable='box')

            laser_data_list = list(laser_data.values())
            rate = len(laser_data) / len(data)
            ros_frame = math.floor((int(frame_num)-1)*rate)
            laser_x = []
            laser_y = []

            for point in laser_data_list[ros_frame]:
                laser_x.append(point[0]*scale/100)
                laser_y.append(point[1]*scale/100)
            plt.plot(laser_x, laser_y, 'ro', markersize=2)
            #print(len(laser_x), len(laser_y))

            for object in frame_data:
                x = object['projected_point'][0]
                y = object['projected_point'][1]
                plt.plot(x*scale, y*scale, 'bo')
                plt.text(x*scale, y*scale, f'{object["class"]}, {object["cam_id"]}')

            for ob_name in cluster_centers[frame_num].keys():
                for id in cluster_centers[frame_num][ob_name].keys():
                    x = cluster_centers[frame_num][ob_name][id][0]
                    y = cluster_centers[frame_num][ob_name][id][1]
                    plt.plot(x*scale, y*scale, 'co')
                    plt.text(x*scale, y*scale, f"id{id} {ob_name}")

            if int(frame_num) > 4:
                for object_name in predictions.keys():
                    for object_id in predictions[object_name].keys():
                        x = predictions[object_name][object_id]['actual_pred'][0]
                        y = predictions[object_name][object_id]['actual_pred'][1]
                        plt.plot(x*scale, y*scale, 'go')
                        plt.text(x*scale, y*scale, f"id{object_id} {object_name}")

            plt.waitforbuttonpress()

    return cluster_centers

if __name__ == "__main__":
    with open(os.path.join(PROJECTED_POINTS_DIR, TEST_NAME + "_det" + ".json"), "r") as json_file:
        projected_data = json.load(json_file)

    with open(os.path.join(LASER_DIR, TEST_NAME + ".json"), "r") as json_file:
        laser_data = json.load(json_file)

    cluster_centers = clustering(projected_data, laser_data)

    with open(os.path.join(PROJECTED_CLUSTER_CENTERS_DIR, TEST_NAME + "_det" + ".json"), "w") as json_file:
        json.dump(cluster_centers, json_file, ensure_ascii=False, indent=4)