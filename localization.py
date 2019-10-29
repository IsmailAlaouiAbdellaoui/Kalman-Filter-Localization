import numpy as np
import math


class Localization:
    def __init__(self, min_timediff_sec, model_var=(0.05, 0.05, 0.05), sensor_var=(0.05 , 0.05, 0.05)):
        self.min_timediff_sec = min_timediff_sec
        self.model_var = model_var
        self.R = np.array([[model_var[0], 0, 0],
                      [0, model_var[1], 0],
                      [0, 0, model_var[2]]], dtype=np.float)
        self.sensor_var = sensor_var
        self.Q = np.array([[sensor_var[0], 0, 0],
                           [0, sensor_var[1], 0],
                           [0, 0, sensor_var[2]]], dtype=np.float)

    def predict(self, prev_pos, prev_pos_cov, u, sensor_values, features):
        delta_t = self.min_timediff_sec
        heading = prev_pos[2]
        predicted_heading = 0
        A = np.eye(3, dtype=np.float)
        B = np.array([[delta_t * math.cos(heading), 0],
                      [-delta_t * math.sin(heading), 0],
                      [0, delta_t]], dtype=np.float)
        C = np.eye(3, dtype=np.float)
        #comptations could be simplified but if we want to change the matrices A, B, C the general
        #method is always working
        #prediction
        pos_pred = np.dot(A, prev_pos) + np.dot(B, u)
        if pos_pred[2] > math.pi * 2:
            pos_pred[2] -= math.pi * 2
        elif pos_pred[2] < -math.pi * 2:
            pos_pred[2] += math.pi * 2
        predicted_heading = pos_pred[2] * 180/math.pi
        pos_cov_pred = np.dot(np.dot(A, prev_pos_cov), np.transpose(A)) + self.R
        if sensor_values.__len__() == 0:
            return (pos_pred, pos_cov_pred, predicted_heading) # If we have no measurements we only use the prediction from the model
        else:
            #converting sensor values to x,y,heading
            z = np.zeros(shape=(3, 1), dtype=np.float)
            for value in sensor_values:
                #get feature position through signature
                x_feature = features[int(round(value[2]))][0]
                y_feature = features[int(round(value[2]))][1]
                # get feature angle in the global frame
                y = y_feature + value[0] * math.sin(value[1] + pos_pred[2])
                x = x_feature - value[0] * math.cos(value[1] + pos_pred[2])
                #print('pred', math.degrees(value[1] + pos_pred[2]), value[0] * math.sin(value[1] + pos_pred[2]), -value[0] * math.cos(value[1] + pos_pred[2]))
                z[0] += x
                z[1] += y
                predicted_heading += value[0] * 180/math.pi

            z /= float(sensor_values.__len__())
            predicted_heading /= sensor_values.__len__() # average predicted heading of all sensor values
            # Just from the bearing and distance we have no information of the heading (more heading value can cause the
            # same measurements, thus we are going to use the value predicted from the motion model)
            z[2] = pos_pred[2]
            #correction
            a = np.linalg.inv(np.dot(np.dot(C, pos_cov_pred), np.transpose(C)) + self.Q)
            K = np.dot(np.dot(pos_cov_pred, np.transpose(C)), a)

            pos_corr = pos_pred + np.dot(K, (z - np.dot(C, pos_pred)))
            pos_cov_corr = np.dot((np.eye(3, dtype=np.float) - np.dot(K, C)), pos_cov_pred)

            return (pos_corr, pos_cov_corr, predicted_heading)


