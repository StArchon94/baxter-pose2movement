#!/usr/bin/env python3.8
from urllib.request import AbstractDigestAuthHandler
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
        R[1, 1] = np.cos(self.radius)
        R[1, 2] = -np.sin(self.radius)
        R[2, 1] = np.sin(self.radius)
        R[2, 2] = np.cos(self.radius)

        return R


class Ry(RorationMatrix):
    @property
    def matrix(self):
        R = np.eye(3)
        R[0, 0] = np.cos(self.radius)
        R[0, 2] = np.sin(self.radius)
        R[2, 0] = -np.sin(self.radius)
        R[2, 2] = np.cos(self.radius)

        return R


class Rz(RorationMatrix):
    @property
    def matrix(self):
        R = np.eye(3)
        R[0, 0] = np.cos(self.radius)
        R[0, 1] = -np.sin(self.radius)
        R[1, 0] = np.sin(self.radius)
        R[1, 1] = np.cos(self.radius)

        return R


class Pose2Joint:
    def __init__(self) -> None:
        self.depth_raw = None
        self.cv_bridge = CvBridge()
        # hard-coded camera
        H = 240
        W = 424
        K = [308.5246276855469, 0.0, 207.89334106445312, 0.0, 308.5497741699219, 118.29705810546875, 0.0, 0.0, 1.0]
        self.camera = Camera(H, W, K)

        self.choice = 2

        # self.valid_flag = False
        self.valid_arm_poses = None
        self.valid_depths = None

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

    @staticmethod
    def check_valid_depth(depth, range=[0.0, 2.0]):
        depth = depth / 1000
        return (depth > range[0] and depth < range[1])

    def depth_callback(self, data):
        self.depth_raw = self.cv_bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')

    def pose_callback(self, data):
        if self.depth_raw is None:
            print('no depth')
            return
        poses = np_bridge.to_numpy_i64(data)
        green_object_idx = np.argwhere(poses[:, -1, 0] == 1)
        if not len(green_object_idx):
            return
        green_object_idx = green_object_idx[0][0]
        green_object_pose = poses[green_object_idx].squeeze()
        arm_poses = green_object_pose[5:11, :]

        # make sure we start publishing with a valid arm pose
        if self.valid_arm_poses is None:
            if np.all(arm_poses > 0):
                self.valid_arm_poses = arm_poses
            else:
                print('no valid poses yet')
                return

        # update last_valid_arm_poses
        for i in range(6):
            if np.all(arm_poses[i,:] > 0):
                self.valid_arm_poses[i,:] = arm_poses[i,:]

        # make sure there are depth valid for all joints
        if self.valid_depths is None:
            valid_depths = -np.ones(6)
            for i in range(6):
                joint_coord_2d = self.valid_arm_poses[i,:]
                joint_depth = self.depth_raw[joint_coord_2d[1], joint_coord_2d[0]]
                if self.check_valid_depth(joint_depth):
                    valid_depths[i] = joint_depth
            if np.all(valid_depths > 0):
                self.valid_depths = valid_depths
            else:
                print(valid_depths)
                print('not all depths are valid yet')
                return

        # update valid depths
        filtered_depth = self.depth_raw.copy()
        for i in range(6):
            joint_coord_2d = self.valid_arm_poses[i,:]
            joint_depth = filtered_depth[joint_coord_2d[1], joint_coord_2d[0]]
            # joint_depth = joint_depth.copy()
            if not self.check_valid_depth(joint_depth):
                joint_depth = self.valid_depths[i]
            else:
                self.valid_depths[i] = joint_depth
            filtered_depth[joint_coord_2d[1], joint_coord_2d[0]] = joint_depth

        left_shoulder = self.valid_arm_poses[0,:]
        right_shoulder = self.valid_arm_poses[1,:]
        left_elbow = self.valid_arm_poses[2,:]
        right_elbow = self.valid_arm_poses[3,:]
        left_wrist = self.valid_arm_poses[4, :]
        right_wrist = self.valid_arm_poses[5, :]

        choice = self.choice

        if choice == 1:
            # get 3d coordinates from 2d points and direct scaling
            robot_right_wrist_3d = self.direct_mapping(right_wrist)
            # print('robot_right_wrist_3d')
            # print(robot_right_wrist_3d)
            self.pub_pose.publish(np_bridge.to_multiarray_f64(robot_right_wrist_3d))

        if choice == 2:
            # get 3d coordiates from depth sensor
            coord_3d_from_camera = self.camera.reconstruct(depth=filtered_depth)
            right_wrist_3d_from_camera = coord_3d_from_camera[right_wrist[1], right_wrist[0]]

            print('right wrist 2d')
            print(right_wrist)

            print('camera frame right_wrist_3d')
            print(right_wrist_3d_from_camera)

            # R_from_camera_to_world = np.eye(3)
            # t_from_caemra_to_world = np.zeros((3,1))
            R_from_camera_to_world = Ry(22 + 90).matrix
            t_from_caemra_to_world = np.array([0.0947, 0, .817])[:, np.newaxis]
            coord_3d_from_world = R_from_camera_to_world[np.newaxis, np.newaxis, :] @ \
                coord_3d_from_camera[:, :, :, np.newaxis] + t_from_caemra_to_world[np.newaxis, np.newaxis, :]

            # print(coord_3d_from_world.shape)

            right_wrist_3d = coord_3d_from_world[right_wrist[1], right_wrist[0]]

            print('world frame right_wrist_3d')
            print(right_wrist_3d)

            robot_right_wrist_3d = self.mirroring(right_wrist_3d)

            print('robot right wrist 3d')
            print(robot_right_wrist_3d)

            self.pub_pose.publish(np_bridge.to_multiarray_f64(robot_right_wrist_3d))

        if choice == 3:
            # output 3d coords for two arms
            coord_3d_from_camera = self.camera.reconstruct(depth=filtered_depth)
            R_from_camera_to_world = np.eye(3)
            t_from_caemra_to_world = np.zeros((3, 1))
            coord_3d_from_world = R_from_camera_to_world[np.newaxis, np.newaxis, :] @ \
                coord_3d_from_camera[:, :, :, np.newaxis] + t_from_caemra_to_world[np.newaxis, np.newaxis, :]

            coords_3d_list = []
            for i in range(5, 11):
                coord_2d = green_object_pose[i, :]
                coord_3d = coord_3d_from_world[coord_2d[1], coord_2d[0]]
                coords_3d_list.append(coord_3d)

            coords_3d = np.array(coords_3d_list).squeeze()
            self.pub_pose.publish(np_bridge.to_multiarray_f64(coords_3d))


if __name__ == '__main__':
    Pose2Joint()
