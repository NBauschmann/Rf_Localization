import rf
import rf_tools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from os import path

wp_filename_rel_path = path.relpath('Waypoints/wp_list_2018_08_23_test.txt')

x0 = [0, 0, 0]
xn = [200, 200, 0]
dxdyda = [50, 50, 0]

rf_tools.wp_generator(wp_filename_rel_path, x0, xn, dxdyda, 5, True)   # 2 is time of measurement

