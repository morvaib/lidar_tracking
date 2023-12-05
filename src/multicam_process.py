import json
import os

import preprocess as preprocesser
import tracking as tracker
import interpolation as interpolater
import camera_merger as merger
import homography_project as projecter
import clustering_and_tracking as finalizer

ROOT_DIR = os.path.abspath("./")

ARUCO_PATH = os.path.join(ROOT_DIR, "aruco_templates/multicam")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
FILTERED_DIR = os.path.join(ROOT_DIR, "filtered")
TRACKED_DIR = os.path.join(ROOT_DIR, "tracked")
PROJECTED_DIR = os.path.join(ROOT_DIR, "projected")
PROJECTED_POINTS_DIR = os.path.join(PROJECTED_DIR, "points")
PROJECTED_CLUSTER_CENTERS_DIR = os.path.join(PROJECTED_DIR, "cluster_centers")

TEST_NAME = "04_03_17_11_sync_det"

def process_all_comdata(filenames):
    input_path = os.path.join(OUTPUT_DIR, TEST_NAME)
    filtered_path = os.path.join(FILTERED_DIR, TEST_NAME)
    tracked_path = os.path.join(TRACKED_DIR, TEST_NAME)
    all_data = []
    if not os.path.exists(filtered_path):
        os.makedirs(filtered_path)
    if not os.path.exists(tracked_path):
        os.makedirs(tracked_path)
    for filename in filenames:
        cam_num = int(filename[3])
        with open(os.path.join(input_path, filename), "r") as json_file:
            com_data = json.load(json_file)

        preprocessed_data = preprocesser.filter_all(com_data)

        with open(os.path.join(filtered_path, filename), "w") as json_file:
            json.dump(preprocessed_data, json_file, ensure_ascii=False, indent=4)

        tracked_data = tracker.track_comdata(preprocessed_data)
        interpolated_data = interpolater.interpolate_all(tracked_data)

        all_data.append([cam_num, interpolated_data])

        with open(os.path.join(tracked_path, filename), "w") as json_file:
            json.dump(interpolated_data, json_file, ensure_ascii=False, indent=4)

    merged_data = merger.make_one_file(all_data)

    with open(os.path.join(TRACKED_DIR, TEST_NAME + ".json"), "w") as json_file:
        json.dump(merged_data, json_file, ensure_ascii=False, indent=4)

    aruco_images = projecter.read_aruco_images()
    homography_matrices = projecter.calculate_all_homography(aruco_images)

    projected_data = projecter.project(merged_data, homography_matrices)

    final_data, clusters = finalizer.multi_tracking(projected_data)

    with open(os.path.join(PROJECTED_POINTS_DIR, TEST_NAME + ".json"), "w") as json_file:
        json.dump(final_data, json_file, ensure_ascii=False, indent=4)

    with open(os.path.join(PROJECTED_CLUSTER_CENTERS_DIR, TEST_NAME + ".json"), "w") as json_file:
        json.dump(clusters, json_file, ensure_ascii=False, indent=4)
    

if __name__ == "__main__":
    _, _, filenames = next(os.walk(os.path.join(OUTPUT_DIR, TEST_NAME)))

    process_all_comdata(filenames)
