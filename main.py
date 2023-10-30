import numpy as np
import cv2
import time
import sys
import torch

sys.path.insert(0, "C:/Users/Artem/PycharmProjects/driveAI/CLRerNet/")

from mmdet.apis import init_detector

from libs.api.inference import inference_one_image
from libs.utils.visualizer import visualize_lanes

from driveAI import Simulation


def rgb_callback(image, data_dict):
    data_dict['rgb_image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))


def depth_callback(image, data_dict, max_depth=0.95):

    depth = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)).astype(np.float32)
    normalized_depth = np.dot(depth[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
    data_dict['depth_image'] = normalized_depth * 1000


def build_extrinsic_matrix(w, h, fov):

    f = w / (2.0 * np.tan(fov * np.pi / 360.0))
    cx = w / 2
    cy = h / 2
    k = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    return k

def camera2d_to_camera3d(u, v, d, inv_k):
    p2d = np.array([u, v, 1])
    p3d = np.dot(inv_k, p2d) * d
    return p3d

if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.get_device_name(0))

    sim = Simulation()
    sim.initialize()

    sensor_data = {}
    for sensor_name in sim.sensors.keys():
        sensor_data[sensor_name] = np.zeros((1640, 590, 4))

    sim.sensors['sensor.camera.rgb'].listen(lambda img: rgb_callback(img, sensor_data))
    sim.sensors['sensor.camera.depth'].listen(lambda img: depth_callback(img, sensor_data))

    cv2.namedWindow('Camera', cv2.WINDOW_AUTOSIZE)
    last_time = time.time()

    model = init_detector("CLRerNet/configs/clrernet/culane/clrernet_culane_dla34_ema.py",
                          "CLRerNet/clrernet_culane_dla34_ema.pth")
    model.cuda()


    x0 = sim.vehicle.get_transform().location.x
    y0 = sim.vehicle.get_transform().location.y
    z0 = sim.vehicle.get_transform().location.z

    sim.spectator.set_transform(sim.vehicle.get_transform())
    debug = sim.world.debug


    image_width = 1640
    image_height = 590
    camera_height = 1.3
    pitch_angle = 0
    yaw_angle = 0
    camera_fov = 100
    camera_position = np.array([0.7, 0, camera_height])


    vector_camera2d_to_world3d = np.vectorize(camera2d_to_camera3d)
    k = build_extrinsic_matrix(image_width, image_height, camera_fov)
    inv_k = np.linalg.inv(k)

    all_lanes_points = None

    src, preds = inference_one_image(model, cv2.resize(cv2.cvtColor(sensor_data['rgb_image'],
                                                                    cv2.COLOR_RGBA2RGB), (1640, 590),
                                                       interpolation=cv2.INTER_AREA))

    t0 = time.time()
    sim.vehicle.set_autopilot(True)
    while True:

        print('{} fps'.format(1 / (time.time() - last_time)))
        last_time = time.time()
        src, preds = inference_one_image(model, cv2.resize(cv2.cvtColor(sensor_data['rgb_image'],
                                                                        cv2.COLOR_RGBA2RGB), (1640, 590),
                                                           interpolation=cv2.INTER_AREA))
        dst = visualize_lanes(src, preds)
        x = sim.vehicle.get_transform().location.x
        y = sim.vehicle.get_transform().location.y
        alpha = np.deg2rad(sim.vehicle.get_transform().rotation.yaw)

        for lane in preds:
            n_points = len(lane)
            lane_points = None
            for i in range(n_points):

                u, v = lane[i]
                u = int(u)
                v = int(v)
                if u < 1600 and 360 < v < 460:
                    p3d = camera2d_to_camera3d(u, v, sensor_data['depth_image'][v, u], inv_k)
                    p3d_world = np.array([p3d[0] * np.cos(alpha) + p3d[2] * np.sin(alpha)
                                                + sim.vehicle.get_transform().location.y - y0,
                                                - p3d[0] * np.sin(alpha) + p3d[2] * np.cos(alpha)
                                                + sim.vehicle.get_transform().location.x - x0])
                    if lane_points is not None:
                        lane_points = np.vstack((p3d_world, lane_points))
                    else:
                        lane_points = p3d_world

            x = lane_points[:, 0]
            xmin = min(x)
            xmax = max(x)
            y = lane_points[:, 1]
            y_p = np.poly1d(np.polyfit(x, y, 2))
            x_p = np.linspace(xmin, xmax, 50)
            lane_points = np.transpose(np.array([x_p, y_p(x_p)]))

            if all_lanes_points is not None:
                all_lanes_points = np.vstack((all_lanes_points, lane_points))
            else:
                all_lanes_points = lane_points


        cv2.imshow('Camera', dst)
        if cv2.waitKey(1) == ord('q') or (time.time() - t0 > 25):
            break

    np.savetxt("points.csv",
               all_lanes_points,
               delimiter=", ",
               fmt='% s')

    for sensor_name in sim.sensors.keys():
        sim.sensors[sensor_name].stop()
    cv2.destroyAllWindows()
    sim.vehicle.destroy()