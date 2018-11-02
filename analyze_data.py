import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import math
import hippocampus_toolbox as hc_tools
from pyquaternion import Quaternion
import tag_class as tc


Old_or_new = 'new'   # 'old' or 'new'


def analyze_data(b_onboard=False, measfilename='path'):
    """
        set meas_type: 
        meas_type = 0: 2D measurement (plot)
        meas_type = 1: 3D measurement with known camera_orientation (set below!)
        meas_type = 2: 3D measurement with unknown camera_orientation
                        -> 6D measurement, using measured quaternion to transform measured position in world frame
    """

    meas_type = 2
    # set this if meas_type = 1
    cam_orientation = Quaternion(0.5, 0.5, 0.5, 0.5)  # quaternion for camera facing wall with big windows

    # unter wasser, z je nach tiefe anpassen (abmessen bis zur wasseroberflaeche)
    offset_camera = [160, 105, 295]

    # ueber wasser mit brotdose
    # offset in z-richtung willkuerlich festgelegt -> tag z-koordinaten relativ dazu anpassen
    offset_camera = [160, 105, 300]

    """ Tag positions and orientations inside the tank """

    """
    frames: 
           - world/map frame: north, east, down, origin in corner 
             between wall 3 and 4 (defined below)
           - camera frame: looking from behind the camera (like a
             photographer), x is right, y is down and z is straight
             ahead
           - tag frame: looking straight at the tag (oriented correctly),
             x is right, y is up and z is towards you (out of the tag)
    Tag orientation:
    rotation from world frame (wf) to tag frame (tf) according to wall 
    for wall closest to big windows (wall 1): q1 = 0.5 - 0.5i - 0.5j + 0.5k
    for wall closest to wave tank (wall 2): q2 = 0 + 0i - 0.707107j + 0.707107k
    for wall closest to computers (wall 3): q3 = 0.5 - 0.5i + 0.5j - 0.5k 
    for wall closest to stairs (wall 4): q4 = 0.707107 - 0.707107i + 0j + 0k
    calculated below
    """
    if Old_or_new == 'old':
        offset_camera = [185.0, 100.0, 15]

        """Orientations of Tags

        rotation from world frame (wf) to tag frame (tf) according to wall 

        tag coordinate system: in tag center, x-axis pointing right, y-axis pointing down, z-axis pointing into tag

        #NEU MACHEN
        for wall closest to big windows (wall 1): q1 = 0.5 + 0.5i + 0.5j + 0.5k
        for wall closest to wave tank (wall 2): q2 = 0 + 0i +0.707107j + 0.707107k
        for wall closest to computers (wall 3): q3 = 0,5 + 0,5i - 0,5j - 0,5k 
        for wall closest to stairs (wall 4): q4 = 0.707107 + 0.707107i + 0j + 0k
        calculated below
        """
        # calculating quaternion for wall 1 (windows)
        tag_w1_orientation_1 = Quaternion(axis=(0, 0, 1.0), degrees=90)
        tag_w1_orientation_2 = Quaternion(axis=(1.0, 0, 0), degrees=90)
        tag_w1_orientation = tag_w1_orientation_1 * tag_w1_orientation_2

        # calculating quaternion for wall 2 (wave tank)
        rotation_w2 = np.array([[-1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]])
        tag_w2_orientation = Quaternion(matrix=rotation_w2)

        # calculating quaternion for wall 3 (computers)
        rotation_w3 = np.array([[0, 0, -1.0], [-1.0, 0, 0], [0, 1.0, 0]])
        tag_w3_orientation = Quaternion(matrix=rotation_w3)

        # calculating quaternion for wall 4 (stairs)
        rotation_w4 = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
        tag_w4_orientation = Quaternion(matrix=rotation_w4)

        # calculating quaternion for tag3:
        tag_3_orientation_1 = Quaternion(axis=[0.0, 1.0, 0.0], degrees=-45)
        tag_3_orientation = tag_w2_orientation * tag_3_orientation_1

        # todo: z-Werte ausmessen und eintragen, sonst macht 3d plot keinen sinn
        tag_0 = tc.Tag(0, np.array([0, 0, 0]), tag_w1_orientation)
        tag_1 = tc.Tag(1, np.array([3820, 445, 0]), tag_w1_orientation)
        tag_2 = tc.Tag(2, np.array([3820, 1390, 0]), tag_w1_orientation)
        tag_3 = tc.Tag(3, np.array([3650, 1830, 40]), tag_3_orientation)

        tags = [tag_0, tag_1, tag_2, tag_3]

    if Old_or_new == 'new':

        # calculating quaternion for wall 1 (windows)
        rotation_w1 = np.array([[0, 0, -1.0], [1.0, 0, 0], [0, -1.0, 0]])
        tag_w1_orientation = Quaternion(matrix=rotation_w1)

        # calculating quaternion for wall 2 (wave tank)
        rotation_w2 = np.array([[-1.0, 0, 0], [0, 0, -1.0], [0, -1.0, 0]])
        tag_w2_orientation = Quaternion(matrix=rotation_w2)

        # calculating quaternion for wall 3 (computers)
        rotation_w3 = np.array([[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]])
        tag_w3_orientation = Quaternion(matrix=rotation_w3)

        # calculating quaternion for wall 4 (stairs)
        rotation_w4 = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
        tag_w4_orientation = Quaternion(matrix=rotation_w4)

        """ Tag Positionen """
        # unter wasser
        tag_0 = tc.Tag(0, np.array([-260, 1323, 368]), tag_w3_orientation)
        tag_1 = tc.Tag(1, np.array([-260, 625, 688]), tag_w3_orientation)
        # alt ueber wasser
        tag_2 = tc.Tag(2, np.array([3820, 1390, 0]), tag_w1_orientation)
        tag_3 = tc.Tag(3, np.array([3650, 1830, 40]), tag_w3_orientation)

        # neu ueber wasser
        tag_4 = tc.Tag(4, np.array([-320, 1320, 190]), tag_w3_orientation)
        tag_5 = tc.Tag(5, np.array([0, 0, 0]), tag_w3_orientation)
        tag_6 = tc.Tag(6, np.array([-260, 950, 297]), tag_w3_orientation)
        tag_7 = tc.Tag(7, np.array([-320, 625, 190]), tag_w3_orientation)

        # unter wasser mit brotdose
        #tags = [tag_0, tag_1, tag_2, tag_3]

        # ueber wasser mit brotdose
        tags = [tag_0, tag_1, tag_2, tag_3, tag_4, tag_5, tag_6, tag_7]


    #if meas_type == 0:
    #    tags = [[0, 0, 0], [3820, 445, 0], [3820, 1390, 0], [3650, 1830, 0]]

    #if meas_type == 1:
    #    tags = [tag_0, tag_1]#, tag_2, tag_3]

    #if meas_type == 2:
    #    tags = [tag_0, tag_1]#, tag_2, tag_3]

    if b_onboard is True:
        meas_data_filename = measfilename
    else:
        meas_data_filename = hc_tools.select_file()

    with open(meas_data_filename, 'r') as meas_file:
        load_description = True
        load_grid_settings = False
        load_meas_data = False
        all_meas_data = []
        # every_wp_list = []  # includes duplicates of wps because of several measurements per wp
        # wp_list = []  # doesn't include duplicates of wps -> better for plotting

        # num_meas_per_wp = 5  # change this if more or less than 5 measurements per waypoint saved.
        plotdata_mat_list = []
        previous_meas = []  # previous measured tags to kick out duplicate measurements because no tag was detected

        for i, line in enumerate(meas_file):

            if line == '### begin grid settings\n':
                print('griddata found')
                load_description = False
                load_grid_settings = True
                load_meas_data = False
                continue
            elif line == '### begin measurement data\n':
                load_description = False
                load_grid_settings = False
                load_meas_data = True
                print('Measurement data found')
                continue
            if load_description:
                #print('file description')
                #print(line)
                continue

            if load_grid_settings and not load_meas_data:
                #print 'reached load_grid_settings'
                grid_settings = map(float, line[:-2].split(' '))
                x0 = [grid_settings[0] + offset_camera[0], grid_settings[1] + offset_camera[1],
                      grid_settings[2] + offset_camera[2]]
                xn = [grid_settings[3] + offset_camera[0], grid_settings[4] + offset_camera[1],
                      grid_settings[5] + offset_camera[2]]
                grid_dxdyda = [grid_settings[6], grid_settings[7], grid_settings[8]]
                timemeas = grid_settings[9]

                data_shape_file = []
                for i in range(3):  # range(num_dof)
                    try:
                        shapei = int((xn[i] - x0[i]) / grid_dxdyda[i] + 1)
                    except ZeroDivisionError:
                        shapei = 1
                    data_shape_file.append(shapei)
                # old: data_shape_file = [int((xn[0]-x0[0]) / grid_dxdyda[0] + 1), int((xn[1]-x0[1]) / grid_dxdyda[1] + 1), int((xn[2]-x0[2]) / grid_dxdyda[2] + 1)]
                print('data shape  = ' + str(data_shape_file))

                # print out
                print('filename = ' + meas_data_filename)
                print('num_of_gridpoints = ' + str(data_shape_file[0] * data_shape_file[1]))
                print('x0 = ' + str(x0))
                print('xn = ' + str(xn))
                print('grid_shape = ' + str(data_shape_file))
                print('steps_dxdyda = ' + str(grid_dxdyda))
                print('timemeas = ' + str(timemeas))

                startx = x0[0]
                endx = xn[0]
                stepx = data_shape_file[0]

                starty = x0[1]
                endy = xn[1]
                stepy = data_shape_file[1]

                startz = x0[2]
                endz = xn[2]
                stepz = data_shape_file[2]

                xpos = np.linspace(startx, endx, stepx)
                ypos = np.linspace(starty, endy, stepy)
                zpos = np.linspace(startz, endz, stepz)

                wp_matx, wp_maty, wp_matz = np.meshgrid(xpos, ypos, zpos)

                # print(xpos)
                # print wp_matx
                wp_vecx = np.reshape(wp_matx, (len(xpos) * len(ypos) * len(zpos), 1))
                wp_vecy = np.reshape(wp_maty, (len(ypos) * len(zpos) * len(xpos), 1))
                wp_vecz = np.reshape(wp_matz, (len(zpos) * len(xpos) * len(ypos), 1))

                wp_mat = np.append(wp_vecx, np.append(wp_vecy, wp_vecz, axis=1), axis=1)
                print wp_mat

            if load_meas_data and not load_grid_settings:

                meas_data_line = map(float, line[:-2].split(' '))

                meas_data_line_list = []

                # reading waypoint data
                meas_data_line_list.append([meas_data_line[0] + offset_camera[0], meas_data_line[1] + offset_camera[1],
                                            meas_data_line[2] + offset_camera[2]])  # wp_x, wp_y, wp_z
                meas_data_line_list.append(
                    [int(meas_data_line[3]), int(meas_data_line[4])])  # wp_num, meas_num of that wp

                # reading tag data
                if len(meas_data_line) > 5:
                    # print ('found at least one tag')
                    num_tags = (len(meas_data_line) - 5) / 8
                    meas_all_tags_list = []

                    for t in range(num_tags):
                        meas_tag_list = []

                        for index in range(8 * t, 8 * (t + 1)):
                            meas_tag_list.append(meas_data_line[5 + index])

                        meas_all_tags_list.append(meas_tag_list)
                    # checking for exact duplicates of measurements (in that moment probably no tag seen
                    # -> saving most recent measurement again)
                    if meas_all_tags_list == previous_meas:
                        # simply don't append to meas_data_line_list
                        pass
                    else:
                        meas_data_line_list.append(meas_all_tags_list)

                    previous_meas = meas_all_tags_list

                # print meas_data_line_list
                all_meas_data.append(meas_data_line_list)

                # print all_meas_data

    # measurement file closed

    # 3D plot
    fig = plt.figure(0)
    pos = 111
    ax = fig.add_subplot(pos, projection='3d')

    # ax.set_xlim([-100, 4000])
    ax.set_xlabel('x-Achse [mm]')

    # ax.axis([-100,4000, -100, 1900, 0,500])#set_ylim([-100, 1900])
    ax.set_ylabel('y-Achse [mm]')

    ax.set_xlim(-100, 4000)
    ax.set_ylim(-100, 2000)
    ax.set_zlim(-100, 1500)

    ax.view_init(elev=-135, azim=45)

    ax.grid()
    ax.set_title('')

    #print wp_mat[:, 0]
    ax.plot(wp_mat[:, 0], wp_mat[:, 1], wp_mat[:, 2], '.', color='#9f9696')  # Grau
    """
    if meas_type == 0:
        for t in range(len(tags)):
            ax.scatter(tags[t][0], tags[t][1], tags[t][2], edgecolor='#cc0000', facecolor='#cc0000')

    if meas_type == 1:
        for t in range(len(tags)):
            ax.scatter(tags[t].get_position_wf()[0], tags[t].get_position_wf()[1], tags[t].get_position_wf()[2], edgecolor='#cc0000', facecolor='#cc0000')
    """

    #for t in range(len(tags)):
        #ax.scatter(tags[t].get_position_wf()[0], tags[t].get_position_wf()[1], tags[t].get_position_wf()[2], color='#660066')

    ax.scatter(tags[4].get_position_wf()[0], tags[4].get_position_wf()[1], tags[4].get_position_wf()[2], color='#660066')
    ax.scatter(tags[6].get_position_wf()[0], tags[6].get_position_wf()[1], tags[6].get_position_wf()[2], color='#660066')
    ax.scatter(tags[7].get_position_wf()[0], tags[7].get_position_wf()[1], tags[7].get_position_wf()[2], color='#660066')

    x_error_all = []
    y_error_all = []
    z_error_all = []

    x_error_tag_0 = []
    y_error_tag_0 = []
    z_error_tag_0 = []

    x_error_tag_1 = []
    y_error_tag_1 = []
    z_error_tag_1 = []

    x_error_tag_2 = []
    y_error_tag_2 = []
    z_error_tag_2 = []

    x_error_tag_3 = []
    y_error_tag_3 = []
    z_error_tag_3 = []

    x_error_tag_4 = []
    y_error_tag_4 = []
    z_error_tag_4 = []

    x_error_tag_5 = []
    y_error_tag_5 = []
    z_error_tag_5 = []

    x_error_tag_6 = []
    y_error_tag_6 = []
    z_error_tag_6 = []

    x_error_tag_7 = []
    y_error_tag_7 = []
    z_error_tag_7 = []

    meas_positions_tag0 = []
    meas_positions_tag1 = []
    meas_positions_tag2 = []
    meas_positions_tag3 = []
    meas_positions_tag4 = []
    meas_positions_tag5 = []
    meas_positions_tag6 = []
    meas_positions_tag7 = []

    for line in range(len(all_meas_data)):

        if len(all_meas_data[line]) > 2:
            num_seen_tags = len(all_meas_data[line][2])
        else:
            num_seen_tags = 0

        # print ('Waypoint, Number of measurement at waypoint: ' + str(all_meas_data[line][1]))
        # print ('seen tags: ' + str(num_seen_tags))

        for tag in range(num_seen_tags):

            tag_id = int(all_meas_data[line][2][tag][0])

            if meas_type == 0:  # 2d plot und messung
                x_meas = float(tags[tag_id][0]) - (float(all_meas_data[line][2][tag][3] * 1000))  # todo: change m to mm in measurement_node.py!
                y_meas = float(tags[tag_id][1]) - (float(all_meas_data[line][2][tag][1] * 1000))
                z_meas = 0
            else:
                if meas_type == 1:  # 3d measurement, known camera position
                    dist_cam_tag_wf = cam_orientation.normalised.rotate(np.array(all_meas_data[line][2][tag][1:4]))  # vector cam-tag in world frame
                    # print all_meas_data[line][2][tag][1:4]
                    # print dist_cam_tag_wf

                if meas_type == 2:  # 3d measurement, unknown camera orientation
                    if Old_or_new == 'new':
                        orientation_cam_tag = Quaternion(all_meas_data[line][2][tag][4:8]).normalised
                    if Old_or_new == 'old':
                        orientation_cam_tag = Quaternion(all_meas_data[line][2][tag][7], all_meas_data[line][2][tag][4], all_meas_data[line][2][tag][5], all_meas_data[line][2][tag][6])
                    #print orientation_cam_tag
                    # orientation_cam_wf = tags[tag_id].get_orientation_wf() * orientation_cam_tag  # same as below
                    orientation_cam_wf = tags[tag_id].convert_orientation_to_wf(orientation_cam_tag)

                    #print orientation_cam_tag
                    dist_cam_tag_cf = np.array(all_meas_data[line][2][tag][1:4]) * 1000   # vector cam-tag in camera frame

                    #print dist_cam_tag_cf
                    position_wf = tags[tag_id].convert_location_to_wf(orientation_cam_tag.normalised, dist_cam_tag_cf)  # vector cam-tag in world frame in mm

                    S_tc = np.array([[1.0, 0, 0], [0, -0.985, 0.174], [0, -0.174, -0.985]])
                    #q_tc = Quaternion(matrix=S_tc)
                    #print q_tc
                    #S_tc = orientation_cam_tag.conjugate.rotation_matrix
                    #print S_tc
                    dist_ct_t = np.dot(S_tc, dist_cam_tag_cf)
                    S_wt = tags[tag_id].get_orientation_wf().rotation_matrix
                    #print S_wt
                    #position_wf = np.dot(S_wt, dist_ct_t)
                    #position_
                    #print position_wf

                #x_meas = float(all_meas_data[line][2][tag][1] * 1000)
                #y_meas = float(all_meas_data[line][2][tag][2] * 1000)
                #z_meas = float(all_meas_data[line][2][tag][3] * 1000)


                #x_meas = float(tags[tag_id].get_position_wf()[0]) - (float(dist_cam_tag_wf[0]))
                #y_meas = float(tags[tag_id].get_position_wf()[1]) - (float(dist_cam_tag_wf[1]))
                #z_meas = float(tags[tag_id].get_position_wf()[2]) - (float(dist_cam_tag_wf[2]))

                x_meas = float(position_wf[0])
                y_meas = float(position_wf[1])
                z_meas = float(position_wf[2])

            if tag_id == 0:
                meas_positions_tag0.append([x_meas, y_meas, z_meas])

            if tag_id == 1:
                meas_positions_tag1.append([x_meas, y_meas, z_meas])

            if tag_id == 2:
                meas_positions_tag2.append([x_meas, y_meas, z_meas])

            if tag_id == 3:
                meas_positions_tag3.append([x_meas, y_meas, z_meas])

            if tag_id == 4:
                meas_positions_tag4.append([x_meas, y_meas, z_meas])

            if tag_id == 5:
                meas_positions_tag5.append([x_meas, y_meas, z_meas])

            if tag_id == 6:
                meas_positions_tag6.append([x_meas, y_meas, z_meas])

            if tag_id == 7:
                meas_positions_tag7.append([x_meas, y_meas, z_meas])

            x_error = (all_meas_data[line][0][0] - x_meas) ** 2
            y_error = (all_meas_data[line][0][1] - y_meas) ** 2
            z_error = (all_meas_data[line][0][2] - z_meas) ** 2

            if tag_id == 0:
                x_error_tag_0.append(x_error)
                y_error_tag_0.append(y_error)
                z_error_tag_0.append(z_error)

            if tag_id == 1:
                x_error_tag_1.append(x_error)
                y_error_tag_1.append(y_error)
                z_error_tag_1.append(z_error)

            if tag_id == 2:
                x_error_tag_2.append(x_error)
                y_error_tag_2.append(y_error)
                z_error_tag_2.append(z_error)

            if tag_id == 3:
                x_error_tag_3.append(x_error)
                y_error_tag_3.append(y_error)
                z_error_tag_3.append(z_error)

            if tag_id == 4:
                x_error_tag_4.append(x_error)
                y_error_tag_4.append(y_error)
                z_error_tag_4.append(z_error)

            if tag_id == 5:
                x_error_tag_5.append(x_error)
                y_error_tag_5.append(y_error)
                z_error_tag_5.append(z_error)

            if tag_id == 6:
                x_error_tag_6.append(x_error)
                y_error_tag_6.append(y_error)
                z_error_tag_6.append(z_error)

            if tag_id == 7:
                x_error_tag_7.append(x_error)
                y_error_tag_7.append(y_error)
                z_error_tag_7.append(z_error)

            # print all_meas_data[line][0]
            # print float(all_meas_data[line][2][tag][3] * 1000)
            # print tag_id, all_meas_data[line][0][0], all_meas_data[line][0][1], x_meas, y_meas, z_meas, x_error, y_error

            x_error_all.append(x_error)
            y_error_all.append(y_error)
            z_error_all.append(z_error)

    print ('average error over all tags in x, y, z: ' + str(
        math.sqrt(sum(x_error_all) / float(len(x_error_all)))) + ', ' + str(
        math.sqrt(sum(y_error_all) / float(len(y_error_all)))) + ', ' + str(
        math.sqrt(sum(z_error_all) / float(len(z_error_all)))))

    """
    print (
        'average error tag 0 in x, y, z: ' + str(
            math.sqrt(sum(x_error_tag_0) / float(len(x_error_tag_0)))) + ', ' + str(
            math.sqrt(sum(y_error_tag_0) / float(len(y_error_tag_0)))) + ', ' + str(
            math.sqrt(sum(z_error_tag_0) / float(len(z_error_tag_0)))))

    print ('average error tag 1 in x, y, z: ' + str(math.sqrt(sum(x_error_tag_1) / float(len(x_error_tag_1)))) + ', ' + str(
        math.sqrt(sum(y_error_tag_1) / float(len(y_error_tag_1)))) + ', ' + str(
        math.sqrt(sum(z_error_tag_1) / float(len(z_error_tag_1)))))
    """

    """
    print ('average error tag 2 in x, y, z: ' + str(math.sqrt(sum(x_error_tag_2) / float(len(x_error_tag_2)))) + ', ' + str(
        math.sqrt(sum(y_error_tag_2) / float(len(y_error_tag_2)))) + ', ' + str(
        math.sqrt(sum(z_error_tag_2) / float(len(z_error_tag_2)))))
    print ('average error tag 3 in x, y, z: ' + str(math.sqrt(sum(x_error_tag_3) / float(len(x_error_tag_3)))) + ', ' + str(
        math.sqrt(sum(y_error_tag_3) / float(len(y_error_tag_3)))) + ', ' + str(
        math.sqrt(sum(z_error_tag_3) / float(len(z_error_tag_3)))))"""

    xs_0 = [x[0] for x in meas_positions_tag0]
    ys_0 = [x[1] for x in meas_positions_tag0]
    zs_0 = [x[2] for x in meas_positions_tag0]
    ax.plot(xs_0, ys_0, zs_0, '.')#, color='#008c00', label='Tag 1')  # , facecolor='#008c00')

    xs_1 = [x[0] for x in meas_positions_tag1]
    ys_1 = [x[1] for x in meas_positions_tag1]
    zs_1 = [x[2] for x in meas_positions_tag1]
    ax.plot(xs_1, ys_1, zs_1, '.', color='#008c00', label='Tag 1')  # , facecolor='#008c00')  # Gruen

    xs_2 = [x[0] for x in meas_positions_tag2]
    ys_2 = [x[1] for x in meas_positions_tag2]
    zs_2 = [x[2] for x in meas_positions_tag2]
    ax.plot(xs_2, ys_2, zs_2, '.', color='#0064de')  # , facecolor='#0064de')  # MuM Blau

    xs_3 = [x[0] for x in meas_positions_tag3]
    ys_3 = [x[1] for x in meas_positions_tag3]
    zs_3 = [x[2] for x in meas_positions_tag3]
    ax.plot(xs_3, ys_3, zs_3, '.', color='#dc214d')  # , facecolor='#dc214d')   # MuM Rot

    xs_0 = [x[0] for x in meas_positions_tag4]
    ys_0 = [x[1] for x in meas_positions_tag4]
    zs_0 = [x[2] for x in meas_positions_tag4]
    #ax.plot(xs_0, ys_0, zs_0, '.')

    xs_0 = [x[0] for x in meas_positions_tag5]
    ys_0 = [x[1] for x in meas_positions_tag5]
    zs_0 = [x[2] for x in meas_positions_tag5]
    ax.plot(xs_0, ys_0, zs_0, '.')

    xs_0 = [x[0] for x in meas_positions_tag6]
    ys_0 = [x[1] for x in meas_positions_tag6]
    zs_0 = [x[2] for x in meas_positions_tag6]
    #ax.plot(xs_0, ys_0, zs_0, '.')

    xs_0 = [x[0] for x in meas_positions_tag7]
    ys_0 = [x[1] for x in meas_positions_tag7]
    zs_0 = [x[2] for x in meas_positions_tag7]
    ax.plot(xs_0, ys_0, zs_0, '.')

    #ax.scatter(float(offset_camera[0] + 2850.0), float(offset_camera[1] + 650.0), float(offset_camera[2] + 0.0), color='#dc214d'  )


    # for legend only
    Tags = Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="#660066",
                  markeredgecolor="#330000")
    wp = Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="#9f9696",
                markeredgecolor="#9f9696")
    #tag1 = Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="#66ff66", markeredgecolor="#009900")
    #tag2 = Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="#6666ff", markeredgecolor="#0000cc")
    #tag3 = Line2D(range(1), range(1), color="white", marker='o', markerfacecolor="#ffb266", markeredgecolor="#994c00")

    #plt.legend((Tags, wp, tag1, tag2, tag3), ('Tag', 'Wegpunkt', 'Messung Tag 0', 'Messung Tag 1'),numpoints=1, loc=1)

    plt.show()

analyze_data(b_onboard=False, measfilename='path')