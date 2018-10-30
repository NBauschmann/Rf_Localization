import rf
import rf_tools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from os import path

wp_filename_rel_path = path.relpath('Waypoints/wp_list_full_tank_d100_d100_Brotdose.txt')

x0 = [100, 500, 0]
xn = [3000, 1500, 0]
dxdyda = [100, 100, 0]

rf_tools.wp_generator(wp_filename_rel_path, x0, xn, dxdyda, 3, True)   # 3 is time of measurement

