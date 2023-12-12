from shapely.geometry import box
import json
import os

ROOT_DIR = os.path.abspath("./")

OUTPUT_DIR = os.path.join(ROOT_DIR, "datastream/yolo7_deepsort/tracked/dogs")
FILTERED_DIR = os.path.join(ROOT_DIR, "datastream/yolo7_deepsort/filtered/dogs")

TEST_NAME = "05_23_sync_det"

MAX_X = 658
MAX_Y = 492
BORDER_THRESHOLD = 10

INTERSECTION_RATIO = 0.75 # max area interection ratio

# size restrictions
SIZES = {'person':{'max_height': 340, 'min_height': 100, 'max_width': 200, 'min_width': 50, 'max_area': 50000, 'min_area': 10000}, 'dog':{'max_height': 200, 'min_height': 40, 'max_width': 200, 'min_width': 40, 'max_area': 50000, 'min_area': 3200}}

MAX_DIST = 15

#check object near the border
def check_border_points(combox, height, width):
    is_border = False
    if combox[0] - width/2 < BORDER_THRESHOLD or combox[0] + width/2 > MAX_X - BORDER_THRESHOLD \
    or combox[1] - height/2 < BORDER_THRESHOLD or combox[1] + height/2 > MAX_Y - BORDER_THRESHOLD:
        is_border = True

    return is_border

#filter by size
def filter_size(com_data):
    filtered_data = {}
    for frame_id, frame_objects in com_data.items():
        frame = []
        for object in frame_objects:
            if object['height'] < SIZES[object['class']]['max_height'] and object['height'] > SIZES[object['class']]['min_height'] \
                and object['width'] < SIZES[object['class']]['max_width'] and object['width'] > SIZES[object['class']]['min_width'] \
                and object['width']*object['height'] < SIZES[object['class']]['max_area'] and object['width']*object['height'] > SIZES[object['class']]['min_area']:

                frame.append(object)

        filtered_data[frame_id] = frame

    return filtered_data

#filter by intersection
def filter_intersect(com_data):
    filtered_data = {}
    for frame_id, frame_objects in com_data.items():
        frame = []
        idx = 0
        for object in frame_objects:
            object_1 = box(object['box_center'][0] - object['width']/2,
                            object['box_center'][1] - object['height']/2,
                            object['box_center'][0] + object['width']/2,
                            object['box_center'][1] + object['height']/2)
            idx2 = 0
            is_valid = True
            for object2 in frame_objects:
                if object != object2 and object['class'] == object2['class']:
                    object_2 = box(object2['box_center'][0] - object2['width']/2,
                                object2['box_center'][1] - object2['height']/2,
                                object2['box_center'][0] + object2['width']/2,
                                object2['box_center'][1] + object2['height']/2)

                    if object_1.intersects(object_2):
                        intersection = object_1.intersection(object_2)

                        if object_1.area <= object_2.area and intersection.area > object_1.area * INTERSECTION_RATIO:
                            frame_objects.remove(object)
                            is_valid = False
                            break

                idx2 += 1

            if is_valid:
                frame.append(object)

            idx += 1

        filtered_data[frame_id] = frame

    return filtered_data



def prefilter(com_data):
    filtered_data = {}
    for frame_id, frame_objects in com_data.items():
        frame = []
        for object in frame_objects:
            is_border_point = check_border_points(object['mask_com'], object['height'], object['width'])

            if object['class'] == 'cat':
                object['class'] = 'dog'

            if object['class'] in ['person', 'dog'] and not is_border_point:
                frame.append(object)
        filtered_data[frame_id] = frame

    return filtered_data


def filter_all(com_data):
    filtered_data = com_data.copy()
    filtered_data = prefilter(filtered_data)
    filtered_data = filter_intersect(filtered_data)
    filtered_data = filter_size(filtered_data)

    return filtered_data

#move the ground point
def move_ground_point(com_data):
    data = {}
    for frame_id, frame_objects in com_data.items():
        frame = []
        for object in frame_objects:
            if object["class"] == "person":
                object["ground_point"][1] = round(object["ground_point"][1] - object["height"]/10)
            elif object["class"] == "dog":
                object["ground_point"][1] = round(object["ground_point"][1] - object["height"]/5)

            frame.append(object)
        data[frame_id] = frame

    return data

if __name__ == "__main__":
    with open(os.path.join(OUTPUT_DIR,  TEST_NAME + ".json"), "r") as json_file:
       com_data = json.load(json_file)

    filtered_data = filter_all(com_data)
    filtered_data = move_ground_point(com_data)

    with open(os.path.join(FILTERED_DIR, TEST_NAME + ".json"), "w") as json_file:
       json.dump(filtered_data, json_file, ensure_ascii=False, indent=4)

    _, _, filenames = next(os.walk(os.path.join(OUTPUT_DIR, TEST_NAME)))

    output_path = os.path.join(FILTERED_DIR, TEST_NAME)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for filename in filenames:
        with open(os.path.join(OUTPUT_DIR, TEST_NAME, filename), "r") as json_file:
            com_data = json.load(json_file)

        filtered_data = prefilter(com_data)
        filtered_data = move_ground_point(filtered_data)
        print(os.path.join(output_path, filename))

        with open(os.path.join(output_path, filename), "w") as json_file:
            json.dump(filtered_data, json_file, ensure_ascii=False, indent=4)
