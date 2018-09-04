#import rospy
import numpy as np
import math


from pyquaternion import Quaternion
#from aprilslam.msg import Apriltags

class Tag(object):
    def __init__(self, tag_id, position_wf, orientation_wf):
        self.__id = tag_id
        self.__position_wf = position_wf
        self.__orientation_wf = orientation_wf

    def get_id(self):
        return self.__id

    def get_position_wf(self):
        return self.__position_wf

    def get_orientation_wf(self):
        return self.__orientation_wf

    def convert_measurement_to_wf(self, quat_cam_tag_x, dist_cam_tag_x):
        dist_cam_tag_tf = quat_cam_tag_x.rotate(dist_cam_tag_x)
        dist_cam_tag_wf = self.__orientation_wf.rotate(dist_cam_tag_tf)

        #absolute_position = np.subtract(self.__position_wf, dist_cam_tag_wf)
        return dist_cam_tag_wf
    
    def convert_orientation_to_wf(self, quat_cam_tag_x):
        """TO DO: check whether this is the right quaternion!!!"""
        
        absolute_orientation = quat_cam_tag_x * self.__orientation_wf
        return absolute_orientation


"""
dist_cam_tag_tf = quat_cam_tag.rotate(dist_cam_tag)
dist_cam_tag_wf = tag_1_orientation.rotate(dist_cam_tag_tf)

absolute_position = np.subtract(tag_1_pose, dist_cam_tag_wf)
"""
