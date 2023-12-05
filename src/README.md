# README #

Development of a multi camera application for tracking moving animals and humans.

## Preparation
- environment for rcnn:
```conda env create -f environment.yml```

- mask rcnn use -> mrcnn dir
- detectron use -> https://github.com/facebookresearch/Detectron

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

