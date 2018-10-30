import cv2
import numpy as np
import pyrealsense2 as rs

config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

pipeline = rs.pipeline()

profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data())

        arr = np.copy(color_image)

        #lower = (120, 50, 0)
        #upper = (255, 90, 10)

        R = [(0, 100), (0, 100), (0, 255)]
        red_range = np.logical_and(R[0][0] < arr[:, :, 0], arr[:, :, 0] < R[0][1])
        green_range = np.logical_and(R[1][0] < arr[:, :, 0], arr[:, :, 0] < R[1][1])
        blue_range = np.logical_and(R[2][0] < arr[:, :, 0], arr[:, :, 0] < R[2][1])
        valid_range = np.logical_and(red_range, green_range, blue_range)

        arr[valid_range] = 200
        arr[np.logical_not(valid_range)] = 0

        cv2.imshow('images', arr)
        cv2.imshow('images2', color_image)
        cv2.waitKey(1)
finally:
    pipeline.stop()