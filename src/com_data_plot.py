import os
import json
import cv2
import time

ROOT_DIR = os.path.abspath("./")

VIDEO_DIR = os.path.join(ROOT_DIR, "videos")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
FILTERED_DIR = os.path.join(ROOT_DIR, "filtered")

TEST_NAME = "04_03_17_11_sync_det/cam2_clip"
#TEST_NAME = "test2_det"

VIDEO_NAME = "04_03_17_11_sync/cam2_clip"
#VIDEO_NAME = "test2"

SLEEP_TIME = False # plot speed change
IS_FILTERED = False # preprocessed data

def calculate_rectangle_points(com, h, w):
    pt1 = (int(com[0] - w/2), int(com[1] - h/2))
    pt2 = (int(com[0] + w/2), int(com[1] + h/2))

    return pt1, pt2


def show_comdata(com_data, video_path):
    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    i = 1
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            for object in com_data[str(i)]:
                pt1, pt2 = calculate_rectangle_points(object['mask_com'], object['height'], object['width'])
                frame = cv2.circle(frame, tuple(object['mask_com']), 5, (0,0,255), -1)
                #frame = cv2.circle(frame, tuple(object['box_center']), 5, (0,0,255), -1)
                frame = cv2.circle(frame, tuple(object['ground_point']), 5, (255,0,0), -1)
                frame = cv2.rectangle(frame, pt1, pt2, (0,0,255), 2)
                cv2.putText(frame, object['class'], tuple(object['mask_com']), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                #cv2.putText(frame, str(object['width']*object['height']), tuple(object['mask_com']), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                #cv2.putText(frame, str(object['height']), tuple(object['mask_com']), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            # Display the resulting frame
            cv2.imshow('Frame', frame)
            if SLEEP_TIME is not None:
                time.sleep(SLEEP_TIME)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

        i+=1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if IS_FILTERED:
        input_path = os.path.join(FILTERED_DIR, TEST_NAME + ".json")

    else:
        input_path = os.path.join(OUTPUT_DIR, TEST_NAME + ".json")

    with open(input_path, "r") as json_file:
        com_data = json.load(json_file)

    video_path = os.path.join(VIDEO_DIR, VIDEO_NAME + ".mp4")

    show_comdata(com_data, video_path)