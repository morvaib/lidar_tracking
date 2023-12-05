# README #

Lidar data based tracking system for humans and dogs

## Environments
- environment for rcnn:
```environment.yml```
- environment for yolov7-deepsort tracking:
```env_yolo7.yml```
- environment for lidar tracking:
```env_lidar_tracking.yml```


## Part created by Lehel Horváth
## Data pipeline
- videos/images -> object detection -> outputs
- outputs -> preprocess -> filtered
- filtered -> tracking/interpolation(/merge if needed) -> tracked
- tracked/aruco_templates -> homography -> projected
- projected -> clustering/multi tracking -> projected
- rosbag -> extract and transform -> laser_coords

## Scripts
- mask_rcnn_test.py/detectron_test.py: test the objection detection outputs
- rcnn.py/detectron.py: detect dogs and persons on input videos
- preprocess.py: filter the detections
- tracking.py: give id-s to the objects on videos
- interpolation.py: interpolate objects on videos
- camera_merger.py: merge multiple camera tracked data into one json
- homography_project.py: project the objects from videos to lab floor
- clustering_and_tracking.py: cluster, give id-s and filter the projected objects
- multicam_process.py: do the pipeline on multiple camera videos (preprocess->tracking->interpolation->merge->homography->multi tracking)
- pickle_to_json_converter.py: utility script
- rosbag_process.py: extract and transform the laser data from rosbags
- laser_plot.py: plot the rosbag laser data
- com_data_plot.py: plot the detection outputs
- tracked_data_plot.py: plot the tracked center of mass data on videos
- projection_plot.py: plot the projected data (optional: with laser data, with cluster centers)
- test_laser.py: tried to cluster laser data (don't work)
- test_program.py: compare results to LiDAR data (3 metrics)
- test_aggregate.py: plot histogram data on multiple test results (avg_dist)

## Part created by Balázs Morvai
## Datastream
- videos -> detecting and tracking -> datastream/yolo7_deepsort/tracked/humans(dogs)
- datastream/yolo7_deepsort/tracked/humans(dogs) -> preprocess -> datastream/yolo7_deepsort/filtered/humans(dogs)
- datastream/yolo7_deepsort/tracked/filtered/humans(dogs) -> object_merger -> datastream/yolo7_deepsort/merged
- datastream/yolo7_deepsort/merged -> homograpy_project -> datastream/yolo7_deepsort/projected/points
- datastream/yolo7_deepsort/projected/points -> clustering -> datastream/yolo7_deepsort/projected/cluster_centers
- datastream/yolo7_deepsort/projected/cluster_centers -> laser data labeling -> lidar_nn/training_data

## Scripts
- yolov7-deepsort-tracking/bridge_wrapper.py: detect and track videos
- preprocess.py: filtering by size, position and moving the ground point of objects
- object_merger.py: merge detections to one file
- clustering_and_tracking_new.py: cluster and track projected points
- laser_labeling.py: assign object to the laser points
- lidar_nn/models.py: neural network models
- lidar_nn/data_generator.py: data generator for training
- lidar_nn/train_model.py: training the models
- lidar_nn/inference.py: predict with the models
- lidar_nn/legdetect_node.py: detect objects based on lidar data
