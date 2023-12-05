import os
import json
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath("./")

LASER_DIR = os.path.join(ROOT_DIR, "laser_coords")

TEST_NAME = "05_03_sync"

LAB_X = 5.4
LAB_Y = 6.27

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
        rosbag_data = json.load(json_file)
    plot_rosbag(rosbag_data)