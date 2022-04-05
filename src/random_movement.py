#!/usr/bin/env python3.6

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

    def __init__(self):
        self.baxter = Baxter()
        self.left_wrist = MaskedInterface(self.baxter.left_arm, [6])
        self.right_wrist = MaskedInterface(self.baxter.right_arm, [6])
        self.cube_arm = None
        self.in_pregrasp = False
        self._semaphore = BoundedSemaphore(1)
        self._pause_semaphore = BoundedSemaphore(1)
        self.paused = False
        self._is_shutdown = True

    def initialize(self):
        self.cube_arm = 'left'
        self._is_shutdown = False
        self.enable()
        self.move_to_neutral()

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
    High Level Interfaces.
    """
    def move_to_pregrasp(self):
        if self.cube_arm == 'left':
            rospy.loginfo('Move to pregrasp left.')
            self._move_arms_to_positions(self.PREGRASP_LEFT_L, self.PREGRASP_LEFT_R, threshold=0.01)
            self.open('right')
        if self.cube_arm == 'right':
            rospy.loginfo('Move to pregrasp right.')
            self._move_arms_to_positions(self.PREGRASP_RIGHT_L, self.PREGRASP_RIGHT_R, threshold=0.01)
            self.open('left')
        self.in_pregrasp = True

    # def move_to_tilt(self, lrot=0., rrot=0.):
    #     self.in_pregrasp = False
    #     if self.cube_arm == 'left':
    #         rospy.loginfo('Move to tilt left.')
    #         lpos = np.asarray(self.TILTED_SWITCH_LEFT_L).copy()
    #         rpos = np.asarray(self.TILTED_SWITCH_LEFT_R).copy()
    #         lpos[-1] += lrot
    #         rpos[-1] += rrot
    #         self._move_arms_to_positions(lpos.tolist(), rpos.tolist(), threshold=0.01)
    #         self.open('right')
    #     if self.cube_arm == 'right':
    #         rospy.loginfo('Move to tilt right.')
    #         lpos = np.asarray(self.TILTED_SWITCH_RIGHT_L).copy()
    #         rpos = np.asarray(self.TILTED_SWITCH_RIGHT_R).copy()
    #         lpos[-1] += lrot
    #         rpos[-1] += rrot
    #         self._move_arms_to_positions(lpos.tolist(), rpos.tolist(), threshold=0.01)
    #         self.open('left')

    # def move_hands_out(self, dist=0.2, arm='both', slow=False):
    #     self.in_pregrasp = False
    #     rospy.loginfo('Move hands out.')
    #     if slow:
    #         dists = [0.05, dist-0.05]
    #     else:
    #         dists = [dist]
    #     for d in dists:
    #         lpos, rpos = None, None
    #         if arm == 'both' or arm == 'right':
    #             rpos = self.baxter.right_arm.solve_ik_xyz_diff(Point(x=0., y=-d, z=0.))
    #         if arm == 'both' or arm == 'left':
    #             lpos = self.baxter.left_arm.solve_ik_xyz_diff(Point(x=0., y=d, z=0.))
    #         self._move_arms_to_positions(lpos, rpos, threshold=0.02)

    # def grasp(self, switch=False):
    #     self.in_pregrasp = False
    #     if switch:
    #         rospy.loginfo('Switch cube arm.')
    #     else:
    #         rospy.loginfo('Grasping cube.')
    #     target = self.switch_target if switch else self.grasp_target
    #     if self.cube_arm == 'right':
    #         self.servo_left.reset()
    #         xyz, a = self.servo_left.state()
    #         target = self.transformer(target, '/left_hand_camera', '/base')
    #         exyz = xyz - target
    #         lpos = list(self.baxter.left_arm.solve_ik_xyz_diff(Point(x=exyz[0], y=exyz[1], z=exyz[2])))
    #         lpos[-1] += a
    #         self._move_arms_to_positions(lpos=lpos)
    #         self.close('left')
    #         if switch:
    #             self.open('right')
    #             self.cube_arm = 'left'
    #     elif self.cube_arm == 'left':
    #         self.servo_right.reset()
    #         xyz, a = self.servo_right.state()
    #         target = self.transformer(target, '/right_hand_camera', '/base')
    #         exyz = xyz - target
    #         rpos = list(self.baxter.right_arm.solve_ik_xyz_diff(Point(x=exyz[0], y=exyz[1], z=exyz[2])))
    #         rpos[-1] += a
    #         self._move_arms_to_positions(rpos=rpos)
    #         self.close('right')
    #         if switch:
    #             self.open('left')
    #             self.cube_arm = 'right'
    #     return a

    # def tilted_switch(self):
    #     if self.cube_arm == 'right':
    #         self.grasp(switch=True)
    #         rpos = self.baxter.right_arm.solve_ik_xyz_diff(Point(x=0., y=-0.2, z=0.1))
    #         self._move_arms_to_positions(rpos=rpos, threshold=0.02)
    #     elif self.cube_arm == 'left':
    #         self.grasp(switch=True)
    #         lpos = self.baxter.left_arm.solve_ik_xyz_diff(Point(x=0., y=0.2, z=0.1))
    #         self._move_arms_to_positions(lpos=lpos, threshold=0.02)

    # def rotate_clockwise(self, arm, angle=np.pi/2., threshold=0.01):
    #     if arm == 'left':
    #         self.rotate_wrist(angle, arm, threshold)
    #     if arm == 'right':
    #         self.rotate_wrist(angle, arm, threshold)

    # def rotate_counterclockwise(self, arm, angle=np.pi/2., threshold=0.01):
    #     if arm == 'left':
    #         self.rotate_wrist(-angle, arm, threshold)
    #     if arm == 'right':
    #         self.rotate_wrist(-angle, arm, threshold)

    # def rotate_flip(self, arm, threshold=0.01):
    #     self.rotate_wrist(np.pi, arm, threshold)

    # def expose_right(self):
    #     rospy.loginfo('Exposing right face.')
    #     self.cube.expose('R', self.cube_arm)
    #     if not self.in_pregrasp:
    #         self.move_to_pregrasp()
    #     if self.cube_arm == 'left':
    #         return
    #     self.grasp(switch=True)
    #     self.move_hands_out(arm='left')
    #     self.move_to_pregrasp()

    # def expose_left(self):
    #     rospy.loginfo('Exposing left face.')
    #     self.cube.expose('L', self.cube_arm)
    #     if not self.in_pregrasp:
    #         self.move_to_pregrasp()
    #     if self.cube_arm == 'right':
    #         return
    #     self.grasp(switch=True)
    #     self.move_hands_out(arm='right')
    #     self.move_to_pregrasp()

    # def expose_top(self):
    #     rospy.loginfo('Exposing top face.')
    #     self.cube.expose('U', self.cube_arm)
    #     if self.cube_arm == 'left':
    #         self.move_to_tilt(np.pi/2., np.pi/2.)
    #     else:
    #         self.move_to_tilt(-np.pi/2., -np.pi/2.)
    #     self.tilted_switch()
    #     self.move_to_pregrasp()

    # def expose_bottom(self):
    #     rospy.loginfo('Exposing bottom face.')
    #     self.cube.expose('D', self.cube_arm)
    #     self.move_to_tilt(-np.pi/2., np.pi/2.)
    #     self.tilted_switch()
    #     self.move_to_pregrasp()

    # def expose_front(self):
    #     rospy.loginfo('Exposing front face.')
    #     self.cube.expose('F', self.cube_arm)
    #     self.move_to_tilt()
    #     self.tilted_switch()
    #     self.move_to_pregrasp()

    # def expose_back(self):
    #     rospy.loginfo('Exposing back face.')
    #     self.cube.expose('B', self.cube_arm)
    #     if self.cube_arm == 'left':
    #         self.move_to_tilt(lrot=np.pi)
    #     if self.cube_arm == 'right':
    #         self.move_to_tilt(rrot=-np.pi)
    #     self.tilted_switch()
    #     self.move_to_pregrasp()

    # def expose(self, face):
    #     f = self.cube(face)
    #     if f == 'U':
    #         self.expose_top()
    #     if f == 'D':
    #         self.expose_bottom()
    #     if f == 'L':
    #         self.expose_left()
    #     if f == 'R':
    #         self.expose_right()
    #     if f == 'F':
    #         self.expose_front()
    #     if f == 'B':
    #         self.expose_back()

    # def _get_pose(self, arm):
    #     if arm == 'left':
    #         return self.baxter.left_arm.get_joint_positions()
    #     else:
    #         return self.baxter.right_arm.get_joint_positions()

    # def _zero_velocity(self, arm):
    #     limb = self.baxter.left_arm if arm == 'left' else self.baxter.right_arm
    #     class ZV(Thread):
    #         def start(self, *args, **kwargs):
    #             self.is_shutdown = False
    #             self.daemon = True
    #             Thread.start(self, *args, **kwargs)

    #         def run(self):
    #             while not self.is_shutdown:
    #                 limb.set_joint_velocities([0,0,0,0,0,0,0])
    #                 time.sleep(0.05)

    #         def stop(self):
    #             self.is_shutdown = True
    #             self.join()
    #     zv = ZV()
    #     zv.start()
    #     return zv

    # def _get_post_grasp_pose(self, arm, d=0.02):
    #     if arm == 'left':
    #         return list(self.baxter.left_arm.solve_ik_xyz_diff(Point(x=0., y=d, z=0.)))
    #     else:
    #         return list(self.baxter.right_arm.solve_ik_xyz_diff(Point(x=0., y=-d, z=0.)))

    # def _get_pose(self, arm):
    #     if arm == 'left':
    #         return list(self.baxter.left_arm.get_joint_positions())
    #     else:
    #         return list(self.baxter.right_arm.get_joint_positions())

    def turn_clockwise(self, face):
        self.expose(face)
        zv_cube = self._zero_velocity(self.cube_arm)
        a = self.grasp()
        free_arm = 'left' if self.cube_arm == 'right' else 'right'
        pos = self._get_post_grasp_pose(free_arm)
        self.rotate_clockwise(free_arm, np.pi / 2. - a)
        pos[-1] = self._get_pose(free_arm)[-1]
        zv_free = self._zero_velocity(free_arm)
        time.sleep(0.5)
        self.open(free_arm)
        time.sleep(0.1)
        zv_free.stop()
        zv_cube.stop()
        if free_arm == 'right':
            self._move_arms_to_positions(rpos=pos, threshold=0.01)
        else:
            self._move_arms_to_positions(lpos=pos, threshold=0.01)
        self.move_hands_out(arm=free_arm, slow=False)

    def turn_counterclockwise(self, face):
        self.expose(face)
        zv_cube = self._zero_velocity(self.cube_arm)
        a = self.grasp()
        free_arm = 'left' if self.cube_arm == 'right' else 'right'
        pos = self._get_post_grasp_pose(free_arm)
        self.rotate_counterclockwise(free_arm, np.pi / 2. + a)
        pos[-1] = self._get_pose(free_arm)[-1]
        zv_free = self._zero_velocity(free_arm)
        time.sleep(0.5)
        self.open(free_arm)
        time.sleep(0.1)
        zv_free.stop()
        zv_cube.stop()
        if free_arm == 'right':
            self._move_arms_to_positions(rpos=pos, threshold=0.01)
        else:
            self._move_arms_to_positions(lpos=pos, threshold=0.01)
        self.move_hands_out(arm=free_arm, slow=False)

    def turn_flip(self, face):
        self.expose(face)
        zv_cube = self._zero_velocity(self.cube_arm)
        a = self.grasp()
        free_arm = 'left' if self.cube_arm == 'right' else 'right'
        pos = self._get_post_grasp_pose(free_arm)
        if self.cube_arm == 'left':
            _threaded_execution(lambda: self.rotate_clockwise(free_arm, np.pi / 2. - a / 2.),
                                lambda: self.rotate_clockwise(self.cube_arm, np.pi / 2. - a / 2.))
        else:
            _threaded_execution(lambda: self.rotate_counterclockwise(free_arm, np.pi / 2. + a / 2.),
                                lambda: self.rotate_counterclockwise(self.cube_arm, np.pi / 2. + a / 2.))
        pos[-1] = self._get_pose(free_arm)[-1]
        zv_free = self._zero_velocity(free_arm)
        time.sleep(0.5)
        self.open(free_arm)
        time.sleep(0.1)
        zv_free.stop()
        zv_cube.stop()
        if free_arm == 'right':
            self._move_arms_to_positions(rpos=pos, threshold=0.01)
        else:
            self._move_arms_to_positions(lpos=pos, threshold=0.01)
        self.move_hands_out(arm=free_arm, slow=False)

    def capture_image(self, rot=None):
        assert rot in [None,'c','cc','2']
        # Hack to ensure that the cube is detectable in images.
        cube_detected = False
        while not cube_detected:
            img_msg = rospy.wait_for_message('/camera/color/image_rect_color', Image, timeout=10.)
            bridge = CvBridge()
            img = bridge.imgmsg_to_cv2(img_msg, "bgr8")
            cube = detect_cube(img)
            if cube is not None:
                img = crop_cube(img, cube)
                cube_detected = True

        h,w = img.shape[:2]
        center = (w / 2, h / 2)
        if rot == 'c':
            M = cv2.getRotationMatrix2D(center, 270, 1.0)
            img = cv2.warpAffine(img, M, (h, w))
        if rot == 'cc':
            M = cv2.getRotationMatrix2D(center, 90, 1.0)
            img = cv2.warpAffine(img, M, (h, w))
        if rot == '2':
            M = cv2.getRotationMatrix2D(center, 180, 1.0)
            img = cv2.warpAffine(img, M, (h, w))
        new_img_msg = bridge.cv2_to_imgmsg(img, "bgr8")
        new_img_msg.header = img_msg.header
        return new_img_msg

    def expose_to_camera(self):
        image_msgs = {}
        if not self.in_pregrasp:
            self.move_to_pregrasp()
        if self.cube_arm == 'right':
            self.expose_right()
        # reset cube orientation. This assumes that this function is only called at the start of solving a new cube.
        self.cube = CubeOrientation()
        self.rotate_clockwise(self.cube_arm)
        image_msgs['U'] = self.capture_image(rot=None) # capture top face
        self.rotate_flip(self.cube_arm, threshold=0.05)
        image_msgs['D'] = self.capture_image(rot=None) # capture bottom face
        self.expose_bottom()
        self.rotate_counterclockwise(self.cube_arm)
        image_msgs['F'] = self.capture_image(rot='cc') # capture front face
        self.rotate_flip(self.cube_arm, threshold=0.05)
        image_msgs['B'] = self.capture_image(rot='cc') # capture back face
        self.expose_bottom()
        self.rotate_clockwise(self.cube_arm)
        image_msgs['R'] = self.capture_image(rot=None) # capture right face
        self.rotate_flip(self.cube_arm, threshold=0.05)
        image_msgs['L'] = self.capture_image(rot='2') # capture left face
        # bridge = CvBridge()
        # for i,image in enumerate(image_msgs.values()):
        #     cv2.imwrite('/home/ripl/im{}.png'.format(i), bridge.imgmsg_to_cv2(image, "bgr8"))
        return image_msgs

    def move_to_pick_up_pose(self, open=True):
        """
        Move to pickup pose.
        """
        self._move_arms_to_positions(self.PICK_UP_LEFT, self.PICK_UP_RIGHT, threshold=0.01)
        if open:
            _threaded_execution(
                lambda: self.open('right'),
                lambda: self.open('left')
            )

    def move_above_cube(self):
        self.in_pregrasp = False
        target = self.switch_target
        if self.cube_arm == 'right':
            servo = self.servo_left
            target = self.transformer(target, '/left_hand_camera', '/base')
            arm = self.baxter.left_arm
        else:
            servo = self.servo_right
            target = self.transformer(target, '/right_hand_camera', '/base')
            arm = self.baxter.right_arm

        xyz, a = servo.state()
        exyz = xyz - target
        pos = list(arm.solve_ik_xyz_diff(Point(x=exyz[0], y=exyz[1], z=0)))
        arm.move_to_joint_positions(pos)

    def pick_up_cube(self):
        """
        Locate and pickup cube
        """

        # clear accumulated detections.
        self.servo_left.reset()
        self.servo_right.reset()

        # set up the picking cube_arm
        while True:
            if self.servo_left.cube_detected():
                self.cube_arm = 'right'
                servo = self.servo_left
                break
            if self.servo_right.cube_detected():
                self.cube_arm = 'left'
                servo = self.servo_right
                break

        # pick up cube
        self.move_above_cube()
        self.grasp(switch=True)

        if self.cube_arm == 'right':
            init_z = self.baxter.right_arm.get_ee_pose()['position'].z
        else:
            init_z = self.baxter.left_arm.get_ee_pose()['position'].z

        self.move_to_pick_up_pose(open=False)
        # save the cube position on the table wrt left arm
        ee = self.baxter.left_arm.get_ee_pose()
        p,q = ee['position'], ee['orientation']
        p = Point(x=p.x, y=p.y, z=init_z + 0.05)
        self.left_cube_init_pos = list(self.baxter.left_arm.solve_ik(p,q))
        ee = self.baxter.right_arm.get_ee_pose()
        p,q = ee['position'], ee['orientation']
        p = Point(x=p.x, y=p.y, z=init_z + 0.05)
        self.right_cube_init_pos = list(self.baxter.right_arm.solve_ik(p,q))
        self.move_hands_out()

        # move to pregrasp
        self.move_to_pregrasp()
        # reset cube orientation
        self.cube = CubeOrientation()

    def put_down_cube(self):
        """
        place cube on table
        """
        if self.cube_arm == 'right':
            self._move_arms_to_positions(rpos=self.right_cube_init_pos, threshold=0.02)
            self.open('right')
        else:
            self._move_arms_to_positions(lpos=self.left_cube_init_pos, threshold=0.02)
            self.open('left')
        self.move_to_pick_up_pose()
        self.move_to_neutral()


def main():
    # TODO: Let's see if this works.
    rospy.init_node('baxter_cube_control')
    c = Controller()
    c.initialize()
    c.cube_arm = 'left'
    c.move_to_pregrasp()

    rospy.loginfo('Move to pregrasp left.')
    c._move_arms_to_positions(c.PREGRASP_LEFT_L, c.PREGRASP_LEFT_R, threshold=0.01)
    c.open('right')

    rospy.loginfo('Move to pregrasp right.')
    c._move_arms_to_positions(c.PREGRASP_RIGHT_L, c.PREGRASP_RIGHT_R, threshold=0.01)
    c.open('left')


if __name__=='__main__':
    main()
