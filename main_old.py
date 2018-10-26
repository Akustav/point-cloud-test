import cv2
import pyrealsense2 as rs
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from skimage.measure import LineModelND, ransac

pipeline = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)


profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print "Depth Scale is: ", depth_scale

align_to = rs.stream.color
align = rs.align(align_to)


def calculate_angle(x0, y0, x1, y1):
    print "points: ({} : {}) ({} : {})".format(x0,y0,x1,y1)
    from math import atan2, degrees, pi
    dx = x1 - x0
    dy = y1 - y0
    rads = atan2(-dy, dx)
    return degrees(rads)

def normalize_array(array):
    return np.interp(array, (array.min(), array.max()), (0, 1))


try:
    while True:
        # Get frameset of color and depth
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
        #images = np.hstack((color_image, depth_image))

        lower = (120, 65, 0)
        upper = (255, 90, 5)
        mask = cv2.inRange(color_image, lower, upper)

        im2, cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # remove basket from data
            depth_copy = np.copy(depth_image)
            cv2.rectangle(depth_copy, (x, y), (x + w, y + h), (0, 0, 0), -1)
            # crop image and get correct depth
            crop = depth_copy[y - h*2: y + h + h*2, x: x + w + w/2]

            crop = np.rot90(crop) * depth_scale
            if crop.size > 0:

                crop_colormap = cv2.applyColorMap(cv2.convertScaleAbs(crop / depth_scale, alpha=0.03), cv2.COLORMAP_JET)
                cv2.imshow('crop_color', crop_colormap)
                cv2.imshow('crop', crop)

                cX, cY = crop.shape

                # find current basket approx distance by center point:
                dist = np.average(depth_image[y:y+h, x:x+w]) * depth_scale


                distance_deviation = 0.2
                max_depth = dist + distance_deviation
                min_depth = dist - distance_deviation

                xyz = []

                for x in range(0, cX, 2):
                    for y in range(0, cY, 2):
                        depth_value = crop[x, y]
                        if min_depth < depth_value < max_depth:
                            #xyz.append([0, y, depth_value])
                            xyz.append([y, depth_value])

                xyz = np.array(xyz)

                #normalize_array()
                #norm_a = normalize_array(xyz[:, 0])
                #norm_b = normalize_array(xyz[:, 1])

                #norm_xy = np.column_stack((norm_a, norm_b))


                #xyz = np.array(norm_xy)

                print "gate dist: {}".format(dist)

                if xyz.size > 0 and True:
                    print "point array min: {}, max: {}".format(xyz[:, 1].min(), xyz[:, 1].max())

                if True and xyz.size > 0:
                    # max deviation in meters
                    try:
                        deviation = 0.1
                        model_robust, inliers = ransac(xyz, LineModelND, min_samples=2, residual_threshold=deviation, max_trials=20)

                        print "model parameters: {}".format(model_robust.params)

                        #vector0 = model_robust.params[0]
                        #vector = model_robust.params[1]

                        #angle1 = np.arctan(vector[1] - vector[0])# * 180 / 3.14
                        #angle2 = np.arctan(vector[0] - vector[1])# * 180 / 3.14

                        #angle3 = np.arctan(vector0[1] - vector0[0])# * 180 / 3.14
                        #angle4 = np.arctan(vector0[0] - vector0[1])# * 180 / 3.14

                        X = np.array([xyz[:, 0]])

                        #print "x: {}".format(X)


                        line_x = np.arange(X.min(), X.max())[:, np.newaxis]

                        #print "lineX: {}".format(line_x)

                        line_y = model_robust.predict(line_x)

                        print line_y[:, 0]

                        #print line_y

                        #calculate_angle(line_y[:, 0, 0][0], line_y[:, 0, 1][0], line_y[:, 0, 0][10],
                        #                      line_y[:, 0, 1][10])

                        #print "line_x: {} , line_y: {}".format(0, line_y[:, 0, 1])

                        #print calculate_angle(1, model_robust.predict_y([1])[0], 0, model_robust.predict_y([0])[0])


                        #print "angles: {} {} {} {}".format(angle1, angle2, angle3, angle4)


                        if True:
                            import matplotlib.pyplot as plt

                            plt.scatter(xyz[:, 0], xyz[:, 1], s=2, c="red", alpha=0.5)

                            plt.plot(line_y[:, 0, 0], line_y[:, 0, 1], color='cornflowerblue', linewidth=1, label='RANSAC regressor')
                            plt.show()

                            #fig = plt.figure()
                            #ax = fig.add_subplot(111, projection='3d')

                            #outliers = inliers == False
                            #ax.scatter(xyz[inliers][:, 0], xy[inliers][:, 1], xyz[inliers][:, 2], c='b', marker='o', label='Inlier data')
                            #ax.scatter(xyz[outliers][:, 0], xyz[outliers][:, 1], xyz[outliers][:, 2], c='r', marker='o', label='Outlier data')

                            #ax.legend(loc='lower left')
                            #plt.show()
                    except Exception as e:
                        print "no fitting pssible " + str(e)


                if False:
                    fig, ax = plt.subplots()
                    ax.scatter(x_axis[:], crop[:, 1], c='red', s=3, label='red', alpha=1, edgecolors='none')

                    ax.legend()
                    ax.grid(True)
                    plt.show()
                    show_plot = False
            else:
                print "No gate detected"

        cv2.imshow('mask', mask)

        cv2.imshow('images', color_image)
        cv2.waitKey(1)
finally:
    pipeline.stop()
