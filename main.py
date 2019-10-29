from map import Map
from map import Robot
import numpy as np
from map import Map
import random
from localization import Localization
import math

#####made by David

num_features = 30
map_size = (500, 600)
min_timediff_sec = 1.0/50.0
#generate features
features1 = []
for i in range(num_features):
        pos_x = np.random.rand() * map_size[1]
        pos_y = np.random.rand() * map_size[0]
        features1.append((pos_x, pos_y))
pos = (300, 250)

features2 = [(200, 300),(250, 300),(300, 300),(350, 300),(400, 300),(450, 300)]
pos = (50, 300)

map = Map(Robot(pos=pos), features2, Localization(min_timediff_sec))
map.simulate()
