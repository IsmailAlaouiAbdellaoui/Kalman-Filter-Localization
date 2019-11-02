import numpy as np
import cv2
import math
import random
import time
from localization import Localization
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt

class Robot:
    def __init__(self, pos, radius=25, map_size=(500, 600, 3)):
        self.pos = pos
        self.prev_pos = pos
        # the radius of the circle of the robot
        self.radius = radius
        # counter clockwise angle counted from the rightmost point (in radian)
        # if both right and left motor speed are negative then it points to the opposite direction
        # sensors should be relative to the original value
        # however collision should take into consideration negaive speed
        self.heading = random.random() * 2 * math.pi

        self.vel = 0
        self.omega = 0

        self.map_size = map_size

        self.color = (255, 0, 0)

        self.sensor_range = 75.0
        self.noise = (0.05, 0.05, 0.005)
        self.counter = 0


    def move(self, deltaT):
        if not (self.vel == 0 and self.omega == 0):
            self.pos = (self.pos[0] + math.cos(self.heading) * self.vel * deltaT,
                        self.pos[1] - math.sin(self.heading) * self.vel * deltaT)

            self.heading += self.omega * deltaT
            if self.heading > math.pi * 2:
                self.heading -= math.pi * 2
            elif self.heading < -math.pi * 2:
                self.heading += math.pi * 2

            #add noise
            self.pos = (self.pos[0] + np.random.normal(0, self.noise[0], size=None), self.pos[1] + np.random.normal(0, self.noise[1], size=None))
            self.heading += np.random.normal(0, self.noise[2], size=None)


class Map:
    def __init__(self, robot, features, localizer, map_size=(500, 600, 3)):
        self.robot = robot
        self.localizer = localizer
        self.features = features
        self.map_size = map_size
        self.image = np.ones(self.map_size, np.uint8) * 255
        self.image_trajectory = np.ones(self.map_size, np.uint8) * 255
        self.thickness = 1
        # the absolute value of velocity should be smaller
        self.vel_limit = 50
        self.omega_limit = 6

        # the time step in seconds
        self.min_timediff_sec = 1.0 / self.vel_limit

        self.counter = 0

        self.sensor_noise = (0.05, 0.05, 0.05)

        self.pos = np.array([[self.robot.pos[0]], [self.robot.pos[1]], [self.robot.heading]], dtype=np.float)
        self.pos_cov = np.array([[self.robot.noise[0], 0, 0], [0, self.robot.noise[1], 0], [0, 0, self.robot.noise[2]]], dtype=np.float)

        self.prev_pos = np.array([[self.robot.pos[0]], [self.robot.pos[1]], [self.robot.heading]], dtype=np.float)
        self.center1 = []
        self.center2 = []
        self.width1 = []
        self.height1 = []
        self.rotation1 = []

    def intersection(self, feature):
        x_diff = feature[0] - self.robot.pos[0]
        y_diff = feature[1] - self.robot.pos[1]
        if x_diff == 0 and y_diff < 0:
            intersection = (np.uint(self.robot.pos[0]), np.uint(self.robot.pos[1] - self.robot.radius))
        elif x_diff == 0 and y_diff > 0:
            intersection = (np.uint(self.robot.pos[0]), np.uint(self.robot.pos[1] + self.robot.radius))
        else:
            angle = math.atan(y_diff / x_diff)
            sign_x = math.copysign(1, x_diff)
            sign_y = math.copysign(1, y_diff)
            intersection = (np.uint(self.robot.pos[0] + sign_x * self.robot.radius * abs(math.cos(angle))), np.uint(self.robot.pos[1] + sign_y * self.robot.radius * abs(math.sin(angle))))
        return intersection

    def draw(self):
        self.draw_trajectory()
        self.image = np.copy(self.image_trajectory)
        self.draw_features()
        self.draw_robot()
        self.draw_sensors()
        self.draw_ellipse()

    # you can also use this to determine the sensor coordinates on the circle line
    def robot_circle_points(self, heading):
        heading_sign_end_point_x = np.uint(
            math.cos(heading) * self.robot.radius + self.robot.pos[0])
        heading_sign_end_point_y = np.uint(
            -math.sin(heading) * self.robot.radius + self.robot.pos[1])
        return (heading_sign_end_point_x, heading_sign_end_point_y)

    def draw_robot(self):
        circ_axes = (self.robot.radius, self.robot.radius)
        cv2.ellipse(self.image, (np.uint(self.robot.pos[0]), np.uint(self.robot.pos[1])), circ_axes, 0, 0, 360,
                    self.robot.color, self.thickness, cv2.LINE_AA)
        cv2.line(self.image, (np.uint(self.robot.pos[0]), np.uint(self.robot.pos[1])),
                 self.robot_circle_points(self.robot.heading), (255, 0, 0),
                 self.thickness+1)

    def draw_features(self):
        for feature in self.features:
            cv2.circle(self.image, (np.uint(feature[0]), np.uint(feature[1])), 2, (255, 0, 0), -1)

    def draw_trajectory(self):
        p1 = int(self.robot.prev_pos[0])
        p2 = int(self.robot.prev_pos[1])
        p3 = int(self.robot.pos[0])
        p4 = int(self.robot.pos[1])
        cv2.line(self.image_trajectory, (p1, p2), (p3, p4), (50, 50, 50), self.thickness)

        p1 = int(self.prev_pos[0])
        p2 = int(self.prev_pos[1])
        p3 = int(self.pos[0])
        p4 = int(self.pos[1])
        cv2.line(self.image_trajectory, (p1, p2), (p3, p4), (0, 0, 255), self.thickness)

        self.prev_pos = self.pos
        self.robot.prev_pos = self.robot.pos

    def draw_sensors(self):
        for feature in self.features:
            distance = math.sqrt((self.robot.pos[0]-feature[0])**2 + (self.robot.pos[1]-feature[1])**2)
            if distance < self.robot.sensor_range and distance > self.robot.radius:
                cv2.line(self.image, self.intersection(feature),
                         (np.uint(feature[0]), np.uint(feature[1])), (0, 255, 0),
                         self.thickness)

    def update_sensor_values(self):
        self.sensor_values = []
        for signature, feature in enumerate(self.features):
            distance = math.sqrt((self.robot.pos[0]-feature[0])**2 + (self.robot.pos[1]-feature[1])**2)
            if distance < self.robot.sensor_range and distance > self.robot.radius:
                x_diff = feature[0] - self.robot.pos[0]
                y_diff = self.robot.pos[1] - feature[1]

                angle = math.atan2(y_diff, x_diff) - self.robot.heading
                #add noise
                distance += np.random.normal(0, self.sensor_noise[0], size=None)
                angle += np.random.normal(0, self.sensor_noise[1], size=None)
                signature += np.random.normal(0, self.sensor_noise[2], size=None)
                self.sensor_values.append((distance, angle, signature))

    def set_robot(self, robot):
        self.robot = robot

    def store_ellipse(self, height, width, rotation, center):
        self.center1.append(np.uint(center[0].item(0)))
        self.center2.append(np.uint(center[1].item(0)))
        self.width1.append(np.uint(width.item(1)))
        self.height1.append(np.uint(height.item(0)))
        self.rotation1.append(rotation)

    def draw_ellipse(self):
        for i in range(len(self.center1)):
            cv2.ellipse(self.image, (self.center1[i], self.center2[i]), (self.width1[i], self.height1[i]),
                        self.rotation1[i], 0, 360, color=(150, 241, 110))
    
    def plot_rmse(self,rmse_array):
        x = np.linspace(1, len(rmse_array), num=len(rmse_array))
        y = rmse_array
        plt.plot(x, y, '.-')
        plt.title('RMSE Plot')
        plt.xlabel('Time Steps')
        plt.show()

    def simulate(self):
        rmse_array = []
        if hasattr(self, 'robot'):
            start = time.clock()
            while (1):
                input = cv2.waitKey(1)
                if input == ord('p'):  # press p to stop
                    self.plot_rmse(rmse_array)
                    cv2.destroyAllWindows()
                    break
                if input == ord('w'):
                    if self.robot.vel < self.vel_limit:
                        self.robot.vel += 1
                elif input == ord('s'):
                    if self.robot.vel >= 1:
                        self.robot.vel -= 1
                elif input == ord('l'):
                    if self.robot.omega < self.omega_limit:
                        self.robot.omega += 1
                elif input == ord('r'):
                    if self.robot.omega >= -self.omega_limit:
                        self.robot.omega -= 1
                elif input == ord('x'):
                    self.robot.vel = 0
                    self.robot.omega = 0

                end = time.clock()
                timediff_ns = end - start
                if timediff_ns > self.min_timediff_sec:
                    self.counter += 1
                    self.robot.move(self.min_timediff_sec)
                    self.update_sensor_values()
                    u = np.array([[self.robot.vel], [self.robot.omega]], dtype=np.float)
                    (pos, pos_cov, predicted_heading) = self.localizer.predict(self.pos, self.pos_cov, u, self.sensor_values, self.features)
                    rmse_array.append(math.sqrt(mse((pos[0],pos[1] ),(self.pos[0],self.pos[1]))))
                    self.pos = pos
                    self.pos_cov = pos_cov
                    self.draw()
                    if self.counter % 100 == 0:
                        self.store_ellipse(pos_cov[0], pos_cov[1], predicted_heading, pos)
                    cv2.imshow('simulator', self.image)
                    start = time.clock()

