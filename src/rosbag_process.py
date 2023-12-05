import rosbag

import os
import subprocess, yaml #required to check bag file contents
import numpy as np
from scipy.spatial.transform import Rotation

import json
import cv2

ROOT_DIR = os.path.abspath("./")
ROSBAG_DIR = os.path.join(ROOT_DIR, "rosbag")

LASER_DIR = os.path.join(ROOT_DIR, "laser_coords")

TEST_NAME = "05_03_sync"

VIDEO_LENGTH = 16 # extracted length (secounds)

SYNCTHREASHOLD = 0.5
WINDOW_LENGTH = 90

ROTATION_ERROR = -0.06 # if transformation is not perfect

def get_transform(bag,from_,to_):

    # get all possible transform messages
    transforms = []
    counter = 0
    for topic, msg, t in bag.read_messages(topics='/tf'):
        transforms += msg.transforms
        counter += 1
        if counter == 50:
            break

    for topic, msg, t in bag.read_messages(topics='/tf_static'):
        transforms += msg.transforms


    # go up the tree
    # from_ should be somewhere up and to_ should be somewhere down
    # eg: from_ is map to_ is laser
    item = to_
    consecutive_transforms = []
    while not item == from_:
        item = next( (x for x in transforms if x.child_frame_id == item))
        consecutive_transforms.append(item.transform)
        item = item.header.frame_id

    # we will need to invert the transforms since we want to go down the stack.
    consecutive_transforms = [[np.array([
                                c.translation.x,
                                c.translation.y,
                                c.translation.z
                                ]),
                                Rotation.from_quat(np.array([
                                c.rotation.x,
                                c.rotation.y,
                                c.rotation.z,
                                c.rotation.w
                                ]))
                                ]
                for c in consecutive_transforms]

    # going up the stack

    r = consecutive_transforms[-1][1].inv()
    t = -1 * r.apply(consecutive_transforms[-1][0])
    for tf_ in reversed(consecutive_transforms[:-1]):
        r = tf_[1].inv()*r
        t = tf_[1].inv().apply(t) - tf_[1].inv().apply(tf_[0])

    return t,r

def frame_to_pic(msg, homography, translation, rotation, intensity_threshold = 0):
    timestamp = msg.header.stamp

    points = []
    for i,(r,intensity) in enumerate(zip(msg.ranges,msg.intensities)):
        if intensity > intensity_threshold:
            phi = msg.angle_min + (i-1)*msg.angle_increment
            x = r*np.cos(phi+ROTATION_ERROR)
            y = r*np.sin(phi+ROTATION_ERROR)
            coords = [x,y,0]
            coords = (rotation.inv().apply(coords-translation))[:2]
            coords = np.array([coords[0],coords[1]]).reshape(-1,1,2)
            coords = cv2.perspectiveTransform(coords, homography)
            #print(coords)
            points.append([coords[0][0][0], coords[0][0][1]])


    # skimage.io.imsave("output/{}_label.jpg".format(msg.header.seq),matrix_image_raw)
    return points

def get_laser_frames(rosbagpath,
                    syncpoint_laser,
                    video_length,
                    homography,
                    translation,
                    rotation,
                    topics = '/rb1_base/front_laser/scan',
                    intensity_threshold = 0,
                    video_frame_per_sec = 30):
    baginfo = yaml.load(subprocess.Popen(['rosbag', 'info', '--yaml', rosbagpath], stdout=subprocess.PIPE).communicate()[0])
    numitems = next((x for x in baginfo['topics'] if x["topic"] == '/rb1_base/front_laser/scan'))["messages"]
    bag = rosbag.Bag(rosbagpath)
    laser_data = {}
    for topic, msg, t in bag.read_messages(topics=topics):
        timestamp = msg.header.stamp.secs + msg.header.stamp.nsecs/1000000000
        #print(timestamp)
        #print(syncpoint_laser)
        if syncpoint_laser < timestamp and timestamp < syncpoint_laser + video_length:
            points = frame_to_pic(
                                            msg,                                       
                                            homography,
                                            translation,
                                            rotation,
                                            intensity_threshold = intensity_threshold)
            
            laser_data[timestamp]= points

    return laser_data

def find_syncpoint(bag):
    already = False
    syncpoint_laser = 0
    window = 0
    for topic, msg, t in bag.read_messages(topics='/rb1_base/front_laser/scan'):
        rangelen = int(np.round(len(msg.ranges)/4.))
        window = int(np.round(WINDOW_LENGTH/msg.scan_time))
        if np.mean(msg.ranges[rangelen:-rangelen]) < SYNCTHREASHOLD:
            timestamp = msg.header.stamp
            syncpoint_laser = timestamp.secs + timestamp.nsecs/1000000000
            already = True
        else:
            if already:
                break
    if syncpoint_laser == 0:
        print("no sync in laser found, terminating")

    return syncpoint_laser, window

def process_rosbag(bag_path):
    bag = rosbag.Bag(bag_path)
    t,r = get_transform(bag,'rb1_base_map','rb1_base_front_laser_link')

    rosmapcoords = np.loadtxt(os.path.join(ROSBAG_DIR, "ros_map_coordinates.csv"), delimiter=";")
    labmapcoords = np.loadtxt(os.path.join(ROSBAG_DIR, "lab_map_coordinates.csv"), delimiter=";")
    homog,_ = cv2.findHomography(rosmapcoords, labmapcoords)
    corners = np.zeros((4,3))
    corners[:,:-1] = rosmapcoords[:4]
    corners = (r.apply(corners) + t)[:,:2]
    xmin = np.min(corners[:,0]) - 1
    xmax = np.max(corners[:,0]) + 1
    ymin = np.min(corners[:,1]) - 1
    ymax = np.max(corners[:,1]) + 1
    print(t)
    syncpoint_laser, window = find_syncpoint(bag)
    #t = [t[1], t[0], t[2]]
    print("syncpoint is {}".format(syncpoint_laser))
    print("window is {} laser frames long".format(window))

    laser_data = get_laser_frames(bag_path, syncpoint_laser, VIDEO_LENGTH, homog, t, r)

    return laser_data
    
if __name__ == "__main__":
    bag_path = os.path.join(ROSBAG_DIR, TEST_NAME + ".bag")
    output = process_rosbag(bag_path)

    with open(os.path.join(LASER_DIR, TEST_NAME + ".json"), "w") as json_file:
        json.dump(output, json_file, ensure_ascii=False, indent=4)