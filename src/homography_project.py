import glob
import os
import json

import cv2
import numpy as np

from test_laser import BORDER_TRESHOLD

ROOT_DIR = os.path.abspath("./")

ARUCO_PATH = os.path.join(ROOT_DIR, "aruco_templates/multicam")
MERGED_DIR = os.path.join(ROOT_DIR, "datastream/yolo7_deepsort/merged")
PROJECTED_DIR = os.path.join(ROOT_DIR, "datastream/yolo7_deepsort/projected")
PROJECTED_POINTS_DIR = os.path.join(PROJECTED_DIR, "points")

TEST_NAME = "05_23_sync_det"
# Labor parameters in meter
LAB_X = 5.4
LAB_Y = 6.27

BORDER_TRESHOLD = 0.1

def check_border(point):
    is_border = False
    if point[0] < BORDER_TRESHOLD or point[1] < BORDER_TRESHOLD or point[0] > LAB_X - BORDER_TRESHOLD or point[1] > LAB_Y - BORDER_TRESHOLD:
        is_border = True

    return is_border

def coord_from_filename(filename):
    x = int(filename[-7])
    y = int(filename[-5])

    return x,y

def calculate_homography(view, aruco_img):
    template_matching_method = cv2.TM_SQDIFF_NORMED

    src_pts = []
    dst_pts = []

    # Find and match the distance points (lab coordinates in meter) with source points (pixel coordinates)
    for aruco_board in sorted(glob.glob(os.path.join(ARUCO_PATH,"{}_*_*.png".format(view)))):

        x, y = coord_from_filename(aruco_board)
        pt = (x, y)
        dst_pts.append(pt)

        template = cv2.imread(aruco_board, 0)

        w, h = template.shape[::-1]

        res = cv2.matchTemplate(aruco_img, template, template_matching_method)
        in_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = min_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        src_pts.append(top_left)

    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)

    homography_mx, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return homography_mx

def calculate_all_homography(aruco_data):
    homography_matrices = []
    for cam_data in aruco_data:
        view = cam_data[0]
        image = cam_data[1]
        homography_mx = calculate_homography(view, image)
        homography_matrices.append([view, homography_mx])

    return sorted(homography_matrices)

def read_aruco_images():
    images = []
    for pic in glob.glob(os.path.join(ARUCO_PATH, "cam?.jpg")):
        cam_id = int(pic[-5])
        image = cv2.imread(pic, 0)
        images.append([cam_id, image])

    return images

def project(com_data, homography_matrices):
    projected_data = {}
    for frame_id, frame_data in com_data.items():
        projected_data[frame_id] = []
        for object in frame_data:
            img_coord = np.array(object['ground_point']).astype(float).reshape(-1,1,2)
            cam_id = object['cam_id']
            warped_coords = cv2.perspectiveTransform(img_coord, homography_matrices[cam_id-1][1])
            x = warped_coords[0,0,0]
            y = warped_coords[0,0,1]
            is_border = check_border([x,y])
            if not is_border:
                object_info = {'cam_id': cam_id, 'id': object['id'], 'mask_com': object['mask_com'], 'ground_point': object['ground_point'], 'projected_point': [x, y], 'class': object['class'], 'score': object['score'], 'height': object['height'], 'width': object['width']}
                projected_data[frame_id].append(object_info)
                #print(frame_id, object_info)

    return projected_data


if __name__ == "__main__":
    aruco_images = read_aruco_images()
    homography_matrices = calculate_all_homography(aruco_images)

    with open(os.path.join(MERGED_DIR, TEST_NAME + ".json"), "r") as json_file:
        com_data = json.load(json_file)

    projected_data = project(com_data, homography_matrices)

    with open(os.path.join(PROJECTED_POINTS_DIR, TEST_NAME + ".json"), "w") as json_file:
        json.dump(projected_data, json_file, ensure_ascii=False, indent=4)
