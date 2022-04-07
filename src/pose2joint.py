#!/usr/bin/env python3.8
import numpy as np
import rospy
from cv_bridge import CvBridge
from np_bridge import np_bridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Int64MultiArray

from camera import Camera

SIZE20M = 20 * 1024 * 1024

class RorationMatrix:
    # Rotation matrix around x-axis
    def __init__(self, degree) -> None:
        self.degree = degree

    @property
    def radius(self):
        return self.degree / 180.0 * np.pi

    @property
    def matrix(self):
        raise NotImplementedError


class Rx(RorationMatrix):
    @property
    def matrix(self):
        R = np.eye(3)
        R[1,1] = np.cos(self.radius)
        R[1,2] = -np.sin(self.radius)
        R[2,1] = np.sin(self.radius)
        R[2,2] = np.cos(self.radius)

        return R

class Ry(RorationMatrix):
    @property
    def matrix(self):
        R = np.eye(3)
        R[0,0] = np.cos(self.radius)
        R[0,2] = np.sin(self.radius)
        R[2,0] = -np.sin(self.radius)
        R[2,2] = np.cos(self.radius)

        return R

class Rz(RorationMatrix):
    @property
    def matrix(self):
        R = np.eye(3)
        R[0,0] = np.cos(self.radius)
        R[0,1] = -np.sin(self.radius)
        R[1,0] = np.sin(self.radius)
        R[1,1] = np.cos(self.radius)

        return R



class Pose2Joint:
    def __init__(self) -> None:
        self.depth = None
        self.cv_bridge = CvBridge()
        # hard-coded camera
        H = 240
        W = 424
        K = [308.5246276855469, 0.0, 207.89334106445312, 0.0, 308.5497741699219, 118.29705810546875, 0.0, 0.0, 1.0]
        self.camera = Camera(H, W, K)

        self.choice = 2

        rospy.init_node('pose2joint')
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, callback=self.depth_callback, queue_size=1, buff_size=SIZE20M)
        rospy.Subscriber('/tracked_poses', Int64MultiArray, callback=self.pose_callback, queue_size=1)
        self.pub_pose = rospy.Publisher('/robot_poses', Float64MultiArray, queue_size=1)
        rospy.loginfo('Pose2Joint Node is Up!')
        rospy.spin()

    @staticmethod
    def mirroring(human_3d):
        mirror_plane = 1.0
        robot_3d = human_3d
        robot_3d[0] = 2 * mirror_plane - robot_3d[0]

        return robot_3d

    @staticmethod
    def direct_mapping(coord_2d):
        fixed_depth = 0.5
        tform_matrix = np.array([[6.66666667e-03, -3.48952056e-20, -8.00000000e-01],
                                [1.86159941e-18, -1.41509434e-03, 2.50000000e-01],
                                [2.44764673e-18, 8.00669401e-19, 1.00000000e+00]])
        coord_2d_hm = np.concatenate((coord_2d, np.ones(1))).T
        robot_3d_hm = tform_matrix @ coord_2d_hm
        robot_3d_hm = robot_3d_hm / robot_3d_hm[-1]
        robot_3d = np.concatenate((np.ones(1) * fixed_depth, robot_3d_hm[:2]))

        return robot_3d

    def depth_callback(self, data):
        self.depth = self.cv_bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')

    def pose_callback(self, data):
        if self.depth is None:
            return
        poses = np_bridge.to_numpy_i64(data)
        # green_object_idx = np.where(poses[:, -1, 0] == 1)
        green_object_idx = np.argwhere(poses[:, -1, 0] == 1)[0][0]
        green_object_pose = poses[green_object_idx].squeeze()
        # left_shoulder = green_object_pose[5,:]
        # right_shoulder = green_object_pose[6,:]
        # left_elbow = green_object_pose[7,:]
        # right_elbow = green_object_pose[8,:]
        # left_wrist = green_object_pose[9,:]
        right_wrist = green_object_pose[10,:]

        choice = self.choice
        
        if choice == 1:
            # get 3d coordinates from 2d points and direct scaling
            robot_right_wrist_3d = self.direct_mapping(right_wrist)
            print('robot_right_wrist_3d')
            print(robot_right_wrist_3d)
            self.pub_pose.publish(np_bridge.to_multiarray_i64(robot_right_wrist_3d))

        if choice == 2:
            # get 3d coordiates from depth sensor
            if self.depth is None:
                print('no depth')
                self.depth = np.zeros((self.camera.H, self.camera.W))
            coord_3d_from_camera = self.camera.reconstruct(depth=self.depth)
            right_wrist_3d_from_camera = coord_3d_from_camera[right_wrist[1], right_wrist[0]]

            print('camera frame right_wrist_3d')
            print(right_wrist_3d_from_camera)

            # R_from_camera_to_world = np.eye(3)
            # t_from_caemra_to_world = np.zeros((3,1))
            R_from_camera_to_world = Ry(22 + 90).matrix
            t_from_caemra_to_world = np.array([0.0947, 0, .817])[:,np.newaxis]
            coord_3d_from_world = R_from_camera_to_world[np.newaxis, np.newaxis, :] @ \
                 coord_3d_from_camera[:,:,:,np.newaxis] + t_from_caemra_to_world[np.newaxis, np.newaxis, :]

            print(coord_3d_from_world.shape)

            right_wrist_3d = coord_3d_from_world[right_wrist[1], right_wrist[0]]
            
            print('world frame right_wrist_3d')
            print(right_wrist_3d)

            robot_right_wrist_3d = self.mirroring(right_wrist_3d)

            print('robot_right_wrist_3d')
            print(robot_right_wrist_3d)

            self.pub_pose.publish(np_bridge.to_multiarray_f64(robot_right_wrist_3d))

        if self.choice == 3:
            # output 3d coords for two arms
            if self.depth is None:
                print('no depth')
                self.depth = np.zeros((self.camera.H, self.camera.W))
            coord_3d_from_camera = self.camera.reconstruct(depth=self.depth)
            R_from_camera_to_world = np.eye(3)
            t_from_caemra_to_world = np.zeros((3,1))
            coord_3d_from_world = R_from_camera_to_world[np.newaxis, np.newaxis, :] @ \
                 coord_3d_from_camera[:,:,:,np.newaxis] + t_from_caemra_to_world[np.newaxis, np.newaxis, :]

            right_wrist_3d = coord_3d_from_world[right_wrist[1], right_wrist[0]]

            coords_3d_list = []
            for i in range(5, 11):
                coord_2d = green_object_pose[i, :]
                coord_3d = coord_3d_from_world[coord_2d[1], coord_2d[0]]
                coords_3d_list.append(coord_3d)

            coords_3d = np.array(coords_3d_list)
            self.pub_pose.publish(np_bridge.to_multiarray_f64(coords_3d))


if __name__ == '__main__':
    Pose2Joint()
