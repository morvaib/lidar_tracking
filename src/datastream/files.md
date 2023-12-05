# Datastream
- videos -> detecting and tracking -> datastream/yolo7_deepsort/tracked/humans(dogs)
- datastream/yolo7_deepsort/tracked/humans(dogs) -> preprocess -> datastream/yolo7_deepsort/filtered/humans(dogs)
- datastream/yolo7_deepsort/tracked/filtered/humans(dogs) -> object_merger -> datastream/yolo7_deepsort/merged
- datastream/yolo7_deepsort/merged -> homograpy_project -> datastream/yolo7_deepsort/projected/points
- datastream/yolo7_deepsort/projected/points -> clustering -> datastream/yolo7_deepsort/projected/cluster_centers
- datastream/yolo7_deepsort/projected/cluster_centers -> laser data labeling -> lidar_nn/training_data

# Scripts
- yolov7-deepsort-tracking/bridge_wrapper.py: detect and track videos
- preprocess.py: filtering by size, position and moving the ground point of objects
- object_merger.py: merge detections to one file
- clustering_and_tracking_new.py: cluster and track projected points
- laser_labeling.py: assign object to the laser points
- legdetect_node.py: detect objects based on lidar data