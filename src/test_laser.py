import os
import json
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = os.path.abspath("./")

LASER_DIR = os.path.join(ROOT_DIR, "laser_coords")
TEST_DIR = os.path.join(ROOT_DIR, "testing")

TEST_NAME = "05_23_sync"

LAB_X = 5.4
LAB_Y = 6.27

BORDER_TRESHOLD = 7
MAX_DIST = 5

def check_border(point):
    is_border = False
    if point[0] < BORDER_TRESHOLD or point[1] < BORDER_TRESHOLD or point[0] > LAB_X*100 - BORDER_TRESHOLD or point[1] > LAB_Y*100 - BORDER_TRESHOLD:
        is_border = True

    return is_border

def preprocess_laser_data(laser_data):
    filtered_laser_data = {}
    for frame_id, frame_data in laser_data.items():
        filtered_laser_data[frame_id] = []
        for point in frame_data:
            if not check_border(point):
                filtered_laser_data[frame_id].append(point)

    return filtered_laser_data


def cluster_laser_data(laser_data):
    for frame_id, frame_data in laser_data.items():
        has_changed = True
        while has_changed:
            has_changed = False
            for point in frame_data:
                for point2 in frame_data:
                    if point != point2:
                        dist = np.linalg.norm(np.array(point) - np.array(point2))
                        if dist < MAX_DIST:
                            frame_data.remove(point)
                            frame_data.remove(point2)
                            frame_data.append([(point[0]+point2[0])/2,(point[1]+point2[1])/2])
                            has_changed = True
                            break
                if has_changed:
                    break

    return laser_data

def plot_rosbag(rosbag_data):
    scale = 1000
    for frame_points in rosbag_data.values():
        plt.cla()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(0, LAB_X*scale)
        plt.ylim(0, LAB_Y*scale)
        plt.gca().set_aspect('equal', adjustable='box')
        x = []
        y = []
        for point in frame_points:
            x.append(point[0]*10)
            y.append(point[1]*10)
        plt.plot(x, y, 'ro', markersize=2)
        plt.pause(0.01)
    plt.show()

if __name__ == "__main__":
    with open(os.path.join(LASER_DIR, TEST_NAME + ".json"), "r") as json_file:
        laser_data = json.load(json_file)
    
    laser_data = preprocess_laser_data(laser_data)
    laser_data = cluster_laser_data(laser_data)
    plot_rosbag(laser_data)

    with open(os.path.join(TEST_DIR, TEST_NAME + ".json"), "w") as json_file:
        json.dump(laser_data, json_file, ensure_ascii=False, indent=4)