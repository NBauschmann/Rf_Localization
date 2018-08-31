import rf
import rf_tools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from os import path

wp_filename_rel_path = path.relpath('Waypoints/wp_list_2018_08_29_d50_no1.txt')

x0 = [0, 0, 0]
xn = [3000, 1550, 0]
dxdyda = [50, 50, 0]

rf_tools.wp_generator(wp_filename_rel_path, x0, xn, dxdyda, 3, True)   # 5 is time of measurement

