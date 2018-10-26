import pyrealsense2 as rs
import numpy as np
import cv2
import math

from skimage.measure import LineModelND, ransac

pc = rs.pointcloud()

points = rs.points()

pipe = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)


profile = pipe.start(config)


depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

align_to = rs.stream.color
align = rs.align(align_to)
count = 0

def clamp(value, bottom, top):
    return min(max(value, bottom), top)

def calculate_angle(x0, y0, x1, y1):
    print "points: ({} : {}) ({} : {})".format(x0,y0,x1,y1)
    from math import atan2, degrees, pi
    dx = x1 - x0
    dy = y1 - y0
    rads = atan2(-dy, dx)
    return degrees(rads)

try:
    while True:
        frames = pipe.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        try:
            vertex_array = np.asanyarray(pc.calculate(aligned_depth_frame).get_vertices())
        except:
            count += 1
            #print "Failed to get vertexes for {} frames".format(count)

        count = 0

        # Validate that both frames are valid

        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        pc.map_to(color_frame)

        # fetch rectangle for goal
        lower = (100, 40, 0)
        upper = (255, 90, 40)
        mask = cv2.inRange(color_image, lower, upper)

        im2, cnts, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            vertex_map = []

            vertex_map = np.array(vertex_array).reshape(720, 1280)

            # mask out basket
            vertex_map[y:y + h, x:x+w] = 0

            min_x = clamp(x + w/2, 0, 1280)
            max_x = clamp(x + w*3/2, 0, 1280)
            min_y = clamp(y - h * 3 / 2, 0, 720)
            max_y = clamp(y + h * 5 / 2, 0, 720)

            cv2.rectangle(color_image, (min_x, min_y), (max_x, max_y), (0, 255, 255), 2)

            crop_vertex = vertex_map[min_y: max_y, min_x: max_x]

            crop_test = depth_image[min_y: max_y, min_x: max_x]

            if crop_test.size > 0:
                crop_test = cv2.applyColorMap(cv2.convertScaleAbs(crop_test, alpha=0.03), cv2.COLORMAP_JET)
                cv2.imshow('crop_test', crop_test)

            planar_vertex_array = []

            dist = np.average(depth_image[y:y + w, x:x + h]) * depth_scale

            max_dist = dist + 0.4
            min_dist = dist - 0.4

            #print "Dist to goal: {}".format(dist)

            crop_mat_size = crop_vertex.shape

            for vertex_x in range(0, crop_mat_size[0], 5):
                for vertex_y in range(0, crop_mat_size[1], 5):
                    vertex = crop_vertex[vertex_x, vertex_y]
                    vert_x = vertex[1]
                    vert_y = vertex[2]
                    vert_sum = abs(vert_x) + abs(vert_y)
                    vert_dist = math.sqrt(vert_sum)
                    if max_dist > vert_dist > min_dist:
                        planar_vertex_array.append([vert_x, vert_y])

            planar_vertex_array = np.array(planar_vertex_array)

            #print "Vertex map size: {}".format(planar_vertex_array.shape)

            angle = 0

            try:
                if planar_vertex_array.size > 0:
                    deviation = 0.1
                    model_robust, inliers = ransac(planar_vertex_array, LineModelND, min_samples=2, residual_threshold=deviation, max_trials=10)

                    vector = model_robust.params[1]

                    angle = np.arctan2(-vector[1], vector[0]) * 180 / 3.14


                    line_x = np.array([0, 1])
                    line_y = model_robust.predict(line_x)

                    print calculate_angle(line_y[:, 0][0], line_y[:, 1][0], line_y[:, 0][1], line_y[:, 1][1])

                    if True:
                        import matplotlib.pyplot as plt

                        plt.scatter(planar_vertex_array[:, 0], planar_vertex_array[:, 1], s=2, c="red", alpha=0.5)
                        plt.show()
                else:
                    angle = 0

            except Exception as e:
                print "fail"
                print e

            #print "angle: ".format(angle)

        #print points.get_vertices()

        cv2.imshow('images', color_image)
        cv2.waitKey(1)
finally:
    pipe.stop()
