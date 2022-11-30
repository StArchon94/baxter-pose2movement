#!/usr/bin/env python3.8

from threading import BoundedSemaphore, Thread

import numpy as np
import rospy
from np_bridge import np_bridge
from ripl_baxter_interface.ros_interface import Baxter
from std_msgs.msg import Float64MultiArray


def _threaded_execution(*fs):
    ts = []
    for f in fs:
        ts.append(Thread(target=f))
        ts[-1].start()
    for t in ts:
        t.join()


class Coords2Movement:
    """ computes the joint angles from 3D coords, and make baxter move accordingly """
    TIMEOUT = 0.3
    THRESHOLD = 0.002
    RESET_TIMEOUT = 3

    def __init__(self) -> None:
        rospy.init_node('coords2movement', anonymous=True)
        rospy.Subscriber('/robot_poses/all', Float64MultiArray, callback=self.coords_callback, queue_size=1)
        self.baxter = Baxter()
        self.angles = None
        self.last_time = None
        self._semaphore = BoundedSemaphore(1)
        rospy.loginfo('Coords2Movement Node is Up!')
        self.initialize()
        Thread(target=self.move).start()
        rospy.spin()

    def initialize(self):
        self.enable()
        print('moving to neutral pose')
        self.move_to_neutral()
        self.baxter.head.set_pan(0)

        # This executes EndEffectorCommand.CMD_CALIBRATE command
        # _threaded_execution(
        #     lambda: self.baxter.left_gripper.calibrate(),
        #     lambda: self.baxter.right_gripper.calibrate()
        # )

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
        self.move_to_neutral()
        self.disable()

    def move_to_neutral(self):
        self._semaphore.acquire()
        _threaded_execution(
            lambda: self.baxter.right_arm.limb.move_to_neutral(),
            lambda: self.baxter.left_arm.limb.move_to_neutral()
        )
        self._semaphore.release()

    def _move_arms_to_positions(self, lpos=None, rpos=None, timeout=None, threshold=None):
        self._semaphore.acquire()
        if timeout is None:
            timeout = self.TIMEOUT
        if threshold is None:
            threshold = self.THRESHOLD
        if lpos:
            def fl(): return self.baxter.left_arm.move_to_joint_positions(lpos, timeout, threshold)
        if rpos:
            def fr(): return self.baxter.right_arm.move_to_joint_positions(rpos, timeout, threshold)
        if lpos and rpos:
            _threaded_execution(fl, fr)
        elif lpos:
            fl()
        elif rpos:
            fr()
        self._semaphore.release()
        rospy.sleep(0.05)

    def move(self):
        while not rospy.is_shutdown():
            if self.angles is None:
                continue
            t = rospy.Time.now()
            if (t - self.last_time).to_sec() > self.RESET_TIMEOUT:
                self.move_to_neutral()
                self.angles = None
            else:
                self._move_arms_to_positions(*self.angles)

    def compute_angles(self, coords):
        l01 = coords[1] - coords[0]
        l02 = coords[2] - coords[0]
        l24 = coords[4] - coords[2]
        l13 = coords[3] - coords[1]
        l35 = coords[5] - coords[3]

        s0_r = np.sign(np.cross(-l01[:2], l02[:2])) * np.arccos(np.dot(-l01[:2], l02[:2]) / (np.linalg.norm(l01[:2]) * np.linalg.norm(l02[:2])))
        s1_r = np.arctan2(-l02[2], np.linalg.norm(l02[:2]))
        v1 = np.array([0, 0, -1]) + l02[2] / np.dot(l02, l02) * l02
        v2 = l24 + np.dot(l02, -l24) / np.dot(l02, l02) * l02
        e0_r = np.sign(np.dot(np.cross(v1, v2), l02)) * np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        e1_r = np.arccos(np.dot(l02, l24) / (np.linalg.norm(l02) * np.linalg.norm(l24)))

        s0_l = np.sign(np.cross(l01[:2], l13[:2])) * np.arccos(np.dot(l01[:2], l13[:2]) / (np.linalg.norm(l01[:2]) * np.linalg.norm(l13[:2])))
        s1_l = np.arctan2(-l13[2], np.linalg.norm(l13[:2]))
        v1 = np.array([0, 0, -1]) + l13[2] / np.dot(l13, l13) * l13
        v2 = l35 + np.dot(l13, -l35) / np.dot(l13, l13) * l13
        e0_l = np.sign(np.dot(np.cross(v1, v2), l13)) * np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        e1_l = np.arccos(np.dot(l13, l35) / (np.linalg.norm(l13) * np.linalg.norm(l35)))

        return [[0.5 + s0_l, s1_l, e0_l, e1_l, 0, 0, 0], [-0.5 + s0_r, s1_r, e0_r, e1_r, 0, 0, 0]]

    def coords_callback(self, data):
        # 0 - left_shoulder
        # 1 - right_shoulder
        # 2 - left_elbow
        # 3 - right_elbow
        # 4 - left_wrist
        # 5 - right_wrist
        self.last_time = rospy.Time.now()
        coords = np_bridge.to_numpy_f64(data)
        self.angles = self.compute_angles(coords)


if __name__ == '__main__':
    Coords2Movement()
