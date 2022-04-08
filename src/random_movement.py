#!/usr/bin/env python3.8

# For the usage of Baxter class, take a look at
# https://github.com/ripl/baxter-rubiks-cube/blob/master/src/baxter-cube-control/src/baxter_cube_control/control.py

import rospy
import sys
import numpy as np
from enum import IntEnum
from time import sleep
from ripl_baxter_interface.ros_interface import Baxter
from sensor_msgs.msg import Image
from ripl_baxter_interface.core import MaskedInterface
from threading import Thread
from geometry_msgs.msg import (
    Point,
    Quaternion,
    PointStamped
)
# from test_ee.msg import TargetEE
from std_msgs.msg import Float64MultiArray, Int64MultiArray
from np_bridge import np_bridge

# from baxter_cube_control.cube_orientation import CubeOrientation
# from baxter_cube_control.visual_servo import CubeServo
# from baxter_cube_control.transformer import Transformer
# from cube_perception.cube_detection import detect_cube, crop_cube
from tf import transformations as T
import cv2, time
from cv_bridge import CvBridge
from threading import BoundedSemaphore

def _threaded_execution(*fs):
    ts = []
    for f in fs:
        ts.append(Thread(target=f))
        ts[-1].start()
    for t in ts:
        t.join()


class Controller(object):
    PREGRASP_LEFT_L = [0.2699475789229093, 0.2842230803643011, -1.4579529514621827, 1.618811812419806, 0.23180537362816822, 0.9775378517825944, -1.4657076440241334]
    PREGRASP_LEFT_R = [-0.2889039180698922, 0.5476051508022153, 1.7810275944305776, 1.350482131374073, -0.5886023531422366, 1.3759077016688446, 0.05726520532698143]
    PREGRASP_RIGHT_L = [0.26256126290559545, 0.24701121207618815, -1.5220738454607683, 1.310223342488052, 0.20555097585840093, 1.2910478882176164, 0.09844366921505648]
    PREGRASP_RIGHT_R = [-0.25318954067337146, 0.33672752885684737, 1.522081087278064, 1.5610919676835255, -0.3083764555181082, 1.0274646466742492, 1.5133432385282293]

    TILTED_SWITCH_RIGHT_R = [0.33764582826343326, 0.4782099124403651, 2.541040519301238, 1.5286118856194542, -1.6424405773039326, 1.8440330609510704, 0.8724936483518282]
    TILTED_SWITCH_RIGHT_L = [0.05435638131828849, -0.45672011769997434, -1.6465695206013813, 1.3693962238889714, 0.47169441232747367, 1.295420736346334, 1.0174264998437448]

    TILTED_SWITCH_LEFT_R = [0.0470467006965357, -0.10628035079733787, 1.9854815764035214, 1.328325078514342, -0.9440946512185494, 1.4901171589496818, -0.835388947295626]
    TILTED_SWITCH_LEFT_L = [0.01128643405038187, -0.4425952728852718, -1.4888447227594674, 1.443680338661239, 0.43724975158427276, 1.134369886080808, -0.4650127117266024]

    PICK_UP_LEFT  = [-0.5518495884417776, -0.7577865092153944, -0.711767085578832, 1.5428011774157548, 0.6055389160177671, 0.9901845985800346, 1.0259808724942042]
    PICK_UP_RIGHT = [0.5518495884417776, -0.7577865092153944, 0.711767085578832, 1.5428011774157548, -0.6055389160177671, 0.9901845985800346, -1.0259808724942042]


    TIMEOUT = 7.
    THRESHOLD = 0.002
    CUBE_WIDTH = 0.053

    def __init__(self, target_ee_right, target_ee_left):
        self.baxter = Baxter()
        self.left_wrist = MaskedInterface(self.baxter.left_arm, [6])
        self.right_wrist = MaskedInterface(self.baxter.right_arm, [6])
        self.cube_arm = None
        self.in_pregrasp = False
        self._semaphore = BoundedSemaphore(1)
        self._pause_semaphore = BoundedSemaphore(1)
        self.paused = False
        self._is_shutdown = True

        # self.pose = rospy.Subscriber(target_ee_pos_topic, TargetEE, self.pose_callback)
        rospy.Subscriber(target_ee_right, PointStamped, self.pose_right_callback)
        rospy.Subscriber(target_ee_left, PointStamped, self.pose_left_callback)

    def initialize(self):
        self.cube_arm = 'left'
        self._is_shutdown = False
        self.enable()
        rospy.loginfo('Moving to neutral pose')
        self.move_to_neutral()

        self.init_quat_r = self.baxter.right_arm.get_ee_pose()['orientation']
        self.init_quat_l = self.baxter.left_arm.get_ee_pose()['orientation']

        # This executes EndEffectorCommand.CMD_CALIBRATE command
        _threaded_execution(
            lambda: self.baxter.left_gripper.calibrate(),
            lambda: self.baxter.right_gripper.calibrate()
        )
        # self.servo_left   = CubeServo(self.baxter.left_arm, '/cube_detection_left/detection', '/cameras/left_hand_camera', self.CUBE_WIDTH, is_rect=True, frame='/base')
        # self.servo_right  = CubeServo(self.baxter.right_arm, '/cube_detection_right/detection', '/cameras/right_hand_camera', self.CUBE_WIDTH, is_rect=True, frame='/base')
        # self.transformer = Transformer()
        # ee = self.transformer([0,0,0], '/right_gripper', '/right_hand_camera')
        # self.grasp_target  = ee - np.array([0,0,self.CUBE_WIDTH / 6])
        # self.switch_target = ee - np.array([0,0,self.CUBE_WIDTH / 2])
        # self.cube = CubeOrientation()

    """
    Pause, Resume, Enable, Disable, Shutdown
    """
    def pause(self):
        self._pause_semaphore.acquire()
        if not self.paused:
            self._semaphore.acquire()
            self.paused = True
            rospy.loginfo("Pausing controller...")
        self._pause_semaphore.release()

    def resume(self):
        self._pause_semaphore.acquire()
        if self.paused:
            self.paused = False
            self._semaphore.release()
            rospy.loginfo("Resuming controller...")
        self._pause_semaphore.release()

    def enable(self):
        self._semaphore.acquire()
        self.baxter.enable()
        self._semaphore.release()

    def disable(self):
        self._semaphore.acquire()
        self.baxter.disable()
        self._semaphore.release()

    def shutdown(self):
        rospy.loginfo("Shutting down Controller...")
        self._is_shutdown = True
        if self.paused:
            self.resume()
        self.move_to_neutral()
        self.disable()

    """
    Moving Baxter.
    """
    def move_to_neutral(self):
        self._semaphore.acquire()
        _threaded_execution(
            lambda: self.baxter.right_arm.limb.move_to_neutral(),
            lambda: self.baxter.left_arm.limb.move_to_neutral()
        )
        self._semaphore.release()

    def _move_arms_to_positions(self, lpos=None, rpos=None, timeout=None, threshold=None):
        if self._is_shutdown:
            return
        self._semaphore.acquire()
        if timeout is None:
            timeout = self.TIMEOUT
        if threshold is None:
            threshold = self.THRESHOLD
        if lpos:
            fl = lambda: self.baxter.left_arm.move_to_joint_positions(lpos, timeout, threshold)
        if rpos:
            fr = lambda: self.baxter.right_arm.move_to_joint_positions(rpos, timeout, threshold)
        if lpos and rpos:
            _threaded_execution(fl, fr)
        elif lpos:
            fl()
        elif rpos:
            fr()
        self._semaphore.release()
        sleep(0.05)

    def open(self, arm):
        if self._is_shutdown:
            return
        self._semaphore.acquire()
        self.in_pregrasp = False
        if arm == 'left':
            self.baxter.left_gripper.open()
        if arm == 'right':
            self.baxter.right_gripper.open()
        self._semaphore.release()
        sleep(0.5)

    def close(self, arm):
        if self._is_shutdown:
            return
        self._semaphore.acquire()
        self.in_pregrasp = False
        if arm == 'left':
            self.baxter.left_gripper.close()
        if arm == 'right':
            self.baxter.right_gripper.close()
        self._semaphore.release()
        sleep(0.5)

    def rotate_wrist(self, radians, arm='left', threshold=0.01):
        if self._is_shutdown:
            return
        self.in_pregrasp = False
        if arm == 'left':
            w = self.left_wrist
        if arm == 'right':
            w = self.right_wrist

        l = w.get_joint_limits()
        ll = l[0][0]
        ul = l[1][0]
        r = w.get_joint_positions()[0] + radians
        if r > ul:
            r -= 2*np.pi
        if r < ll:
            r += 2*np.pi
        w.move_to_joint_positions([r], threshold=threshold)
        sleep(0.25)


    """
    Callback
    """
    def pose_right_callback(self, poses):
        rpos_ee = Point(*np_bridge.to_numpy_f64(poses))
        rospy.loginfo(f'Received rpos_ee: {rpos_ee}')
        # lpos_ee = poses.left_ee
        # rpos_ee = poses.right_ee

        # jquat_r = self.baxter.right_arm.get_ee_pose()['orientation']
        # jquat_l = self.baxter.left_arm.get_ee_pose()['orientation']
        jpos_r = self.baxter.left_arm.solve_ik(rpos_ee, self.init_quat_r)
        if jpos_r is None:
            rospy.logwarn('[warn] IK solution not found!')
            rospy.loginfo('[info] IK solution not found!')
        # jpos_l = self.baxter.right_arm.solve_ik(lpos_ee, self.init_quat_l)

        # MOVE
        self.baxter.right_arm.move_to_joint_positions(jpos_r)
        # self.baxter.left_arm.move_to_joint_positions(jpos_l)

        rospy.loginfo('pose callback is called!!')

    def pose_left_callback(self, poses):
        pos_ee = Point(*np_bridge.to_numpy_f64(poses))
        rospy.loginfo(f'Received rpos_ee: {pos_ee}')

        jpos = self.baxter.left_arm.solve_ik(pos_ee, self.init_quat_l)
        if jpos is None:
            rospy.logwarn('[warn] IK solution not found!')
            rospy.loginfo('[info] IK solution not found!')

        # MOVE
        self.baxter.left_arm.move_to_joint_positions(jpos)
        # self.baxter.left_arm.move_to_joint_positions(jpos_l)

        rospy.loginfo('pose callback is called!!')


def main():
    # TODO: Let's see if this works.
    rospy.init_node('baxter_cube_control')
    rospy.loginfo('Instantiating controller')
    c = Controller(
        target_ee_right='/robot_poses/right',
        target_ee_left='/robot_poses/left',
    )
    rospy.loginfo('Initializing controller')
    c.initialize()
    # c.move_to_neutral()
    # rospy.loginfo('Moving left arm to pregrasp pos')
    # c._move_arms_to_positions(c.PREGRASP_LEFT_L, c.PREGRASP_LEFT_R, threshold=0.01)
    rospy.loginfo('Spinning controller...')
    rospy.spin()

    # c.cube_arm = 'left'
    # c.move_to_neutral()
    # c._move_arms_to_positions(c.PREGRASP_LEFT_L, c.PREGRASP_LEFT_R, threshold=0.01)


    # Use current orientation
    # jquat_r = c.baxter.right_arm.get_ee_pose()['orientation']
    # jquat_l = c.baxter.left_arm.get_ee_pose()['orientation']

    # rospy.loginfo(f'jquat_r:\n{jquat_r}')
    # rospy.loginfo(f'jquat_l:\n{jquat_l}')


    # jpos_r = c.baxter.right_arm.solve_ik(position, jquat_r)
    # jpos_l = c.baxter.left_arm.solve_ik(position, jquat_l)
    # import time
    # counter = 0
    # while True:
    #     # jpos_r = c.baxter.right_arm.solve_ik_xyz_diff(Point(0, 0, 0.05 * counter))
    #     # jpos_l = c.baxter.left_arm.solve_ik_xyz_diff(Point(0, 0, -0.05 * counter))
    #     jpos_r = c.baxter.right_arm.solve_ik(Point(0, 0, 0.05 * counter))
    #     jpos_l = c.baxter.left_arm.solve_ik(Point(0, 0, -0.05 * counter))

    #     c.baxter.right_arm.move_to_joint_positions(jpos_r)
    #     c.baxter.right_arm.move_to_joint_positions(jpos_l)

    #     pos_r = c.baxter.right_arm.get_ee_pose()['position']
    #     pos_l = c.baxter.left_arm.get_ee_pose()['position']

    #     rospy.loginfo(f'pos_r:\n{pos_r}')
    #     rospy.loginfo(f'pos_l:\n{pos_l}')

    #     time.sleep(0.3)
    #     counter += 1

    # for i in range(10):
    #     jpos_r = c.baxter.right_arm.solve_ik_xyz_diff(Point(0, 0, 0.05 * i))
    #     jpos_l = c.baxter.left_arm.solve_ik_xyz_diff(Point(0, 0, -0.05 * i))

    #     # MOVE
    #     c.baxter.right_arm.move_to_joint_positions(jpos_r)
    #     c.baxter.left_arm.move_to_joint_positions(jpos_l)

    #     pos_r = c.baxter.right_arm.get_ee_pose()['position']
    #     pos_l = c.baxter.left_arm.get_ee_pose()['position']

    #     rospy.loginfo(f'pos_r:\n{pos_r}')
    #     rospy.loginfo(f'pos_l:\n{pos_l}')

    # c.baxter.right_arm.solve_ik_xyz_diff(diff)
    # c.baxter.right_arm.get_ee_pose()  # ee['position'].x


    # rospy.loginfo('Move to pregrasp left.')
    # c._move_arms_to_positions(c.PREGRASP_LEFT_L, c.PREGRASP_LEFT_R, threshold=0.01)
    # c.open('right')

    # rospy.loginfo('Move to pregrasp right.')
    # c._move_arms_to_positions(c.PREGRASP_RIGHT_L, c.PREGRASP_RIGHT_R, threshold=0.01)
    # c.open('left')


if __name__=='__main__':
    main()
