import os
import numpy as np
import json
import matplotlib.pyplot as plt
import math
import copy
import tqdm

ROOT_DIR = os.path.abspath("./datastream/yolo7_deepsort")
PROJECTED_DIR = os.path.join(ROOT_DIR, "projected")
PROJECTED_POINTS_DIR = os.path.join(PROJECTED_DIR, "points")
PROJECTED_CLUSTER_CENTERS_DIR = os.path.join(PROJECTED_DIR, "cluster_centers")
TRAINING_DATA_DIR = ("./lidar_nn/training_data")
LASER_DIR ="./laser_coords"
TEST_NAME = "04_03_17_11_sync"

LAB_X = 5.4
LAB_Y = 6.27
radius = {"dog": 0.35, "person": 0.25}
colors = ["wheat", "limegreen", "skyblue", "slategrey", "lightcoral"]

def check_border(point, limit):
    if point[0] < limit or point[0] > LAB_X * 100 - limit or point[1] < limit or point[1] > LAB_Y * 100 - limit:
        return True
    return False

def calc_center(points):
    if len(points) == 0:
        return [-1, -1]
    sum_x = 0
    sum_y = 0
    for i in range(len(points)):
        sum_x += points[i][0]
        sum_y += points[i][1]
    return  [sum_x/len(points), sum_y/len(points)]

def label_laser_points(laser_data, center_data):
    training_data = {}
    clusters = {}
    clusetring = True
    plot = True
    pbar = tqdm.tqdm(total=len(center_data))

    laser_data_list = list(laser_data.values())
    rate = len(laser_data) / len(center_data)

    for frame_num, frame_data in center_data.items():
        training_data[frame_num] = {"person": {}, "dog": {}}
        clusters[frame_num] = {"person": {}, "dog": {}}
        ros_frame = math.floor((int(frame_num)-1)*rate)
        pbar.update(1)

        #distance based assigning
        if not clusetring:
            for object_name in frame_data.keys():
                for object_id in frame_data[object_name].keys():
                    training_data[frame_num][object_name][object_id] = []
                    for point in laser_data_list[ros_frame]:
                        if np.linalg.norm(np.array(point)/100 - np.array(frame_data[object_name][object_id])) < radius[object_name]:
                            training_data[frame_num][object_name][object_id].append([point[0], point[1]])

        #clustering
        else:
            for object_name in frame_data.keys():
                for object_id in frame_data[object_name].keys():
                    clusters[frame_num][object_name][object_id] = {"center": frame_data[object_name][object_id], "points": []}

            #print(f"before: {clusters[frame_num]}")

            n = 0
            while n < 10:
                n += 1
                for object_name in clusters[frame_num].keys():
                    for object_id in clusters[frame_num][object_name].keys():
                        if int(frame_num) != 2:
                            clusters[frame_num][object_name][object_id]["points"] = []

                for point in laser_data_list[ros_frame]:
                    min_dist = 1000
                    min_dist_id = 0
                    min_dist_name = None
                    for object_name in clusters[frame_num].keys():
                        for object_id in clusters[frame_num][object_name].keys():
                            dist = np.linalg.norm(np.array(clusters[frame_num][object_name][object_id]["center"]) - np.array(point)/100)
                            if dist < min_dist and not check_border(point, 10):
                                min_dist = dist
                                min_dist_name = object_name
                                min_dist_id = object_id
                    if min_dist_name is not None and [int(point[0]), int(point[1])] not in clusters[frame_num][min_dist_name][min_dist_id]["points"]:
                        clusters[frame_num][min_dist_name][min_dist_id]["points"].append([int(point[0]), int(point[1])])

                for object_name in clusters[frame_num].keys():
                    for object_id in clusters[frame_num][object_name].keys():
                        new_center = calc_center(clusters[frame_num][object_name][object_id]["points"])
                        clusters[frame_num][object_name][object_id]["center"] = [new_center[0]/100, new_center[1]/100]

            for object_name in clusters[frame_num].keys():
                for object_id in clusters[frame_num][object_name].keys():
                    training_data[frame_num][object_name][object_id] = copy.deepcopy(clusters[frame_num][object_name][object_id]["points"])

            # for object_name in clusters[frame_num].keys():
            #     for object_id in clusters[frame_num][object_name].keys():
            #         print(f"{object_name}{object_id}: {clusters[frame_num][object_name][object_id]['center']}")

        if plot:
            scale = 1000
            plt.cla()
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.xlim(0, LAB_X*scale)
            plt.ylim(0, LAB_Y*scale)
            plt.text(5000, 5800, frame_num, fontsize=12)
            plt.gca().set_aspect('equal', adjustable='box')
            laser_x = []
            laser_y = []

            for point in laser_data_list[ros_frame]:
                laser_x.append(point[0]*scale/100)
                laser_y.append(point[1]*scale/100)
            plt.plot(laser_x, laser_y, 'ro', markersize=2)

            colors_num = 0
            for object_name in training_data[frame_num].keys():
                for object_id in training_data[frame_num][object_name].keys():
                    x_cent = clusters[frame_num][object_name][object_id]["center"][0]
                    y_cent = clusters[frame_num][object_name][object_id]["center"][1]
                    plt.plot(x_cent*scale, y_cent*scale, 'co')
                    plt.text(x_cent*scale, y_cent*scale, f"id{object_id} {object_name}")

                    x_laser_assigned = []
                    y_laser_assigned = []

                    for point in training_data[frame_num][object_name][object_id]:
                        x_laser_assigned.append(point[0]*scale/100)
                        y_laser_assigned.append(point[1]*scale/100)
                    plt.plot(x_laser_assigned, y_laser_assigned, marker='o', color=colors[colors_num], markersize=2, linestyle="")
                    colors_num +=1
            plt.waitforbuttonpress()

    return training_data

if __name__ == '__main__':
    with open(os.path.join(LASER_DIR, TEST_NAME + ".json"), "r") as json_file:
        laser_data = json.load(json_file)

    with open(os.path.join(PROJECTED_CLUSTER_CENTERS_DIR, TEST_NAME + "_det" + ".json"), "r") as json_file:
        center_data = json.load(json_file)

    training_data = label_laser_points(laser_data, center_data)

    with open(os.path.join(TRAINING_DATA_DIR, TEST_NAME + ".json"), "w") as json_file:
        json.dump(training_data, json_file, ensure_ascii=False, indent=4)