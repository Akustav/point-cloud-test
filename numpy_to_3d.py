import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from skimage.measure import LineModelND, ransac

pipeline = rs.pipeline()

#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

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

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_image = cv2.resize(depth_image, (64, 36))

        dX, dY = depth_image.shape

        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')

        x_arr = []
        y_arr = []
        z_arr = []

        xyz = []

        for x in range(0, dX):
            for y in range(0, dY):
                x_arr.append(x)
                y_arr.append(y)
                z_arr.append(depth_image[x, y])
                vvv = depth_image[x, y] * depth_scale
                if 0.5 < vvv < 1:
                    xyz.append([0, y, vvv])

        x_arr = np.array(x_arr)
        y_arr = np.array(y_arr)
        z_arr = np.array(z_arr) * depth_scale
        xyz = np.array(xyz)

        try :
            model_robust, inliers = ransac(xyz, LineModelND, min_samples=2,
                                       residual_threshold=0.005, max_trials=20)
        except:
            print "Nothing to fit"

        # `origin`, `direction`
        params = model_robust.params

        #print params

        arctan = np.arctan(params[1][2] / params[1][1]) * 180/3.14

        print "angle = " + str(arctan)

        #outliers = inliers == False

        #ax.scatter(xyz[inliers][:, 0], xyz[inliers][:, 1], xyz[inliers][:, 2], c='b',
        #           marker='o', label='Inlier data')
        #ax.scatter(xyz[outliers][:, 0], xyz[outliers][:, 1], xyz[outliers][:, 2], c='r',
        #           marker='o', label='Outlier data')

        #ax.scatter(x_arr, y_arr, z_arr, c="red", marker="x")

        #ax.legend(loc='lower left')
        #plt.show()

        cv2.imshow('Align Example', depth_colormap)
        cv2.waitKey(1)
finally:
    pipeline.stop()
