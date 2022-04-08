#!/usr/bin/env python3.8

import rospy
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from nav_msgs.msg import Path
from np_bridge import np_bridge
from std_msgs.msg import Float64MultiArray, Header


class CoordsViz:
    def __init__(self) -> None:
        rospy.init_node('coords_viz')
        rospy.Subscriber('/robot_poses/all', Float64MultiArray, callback=self.coords_callback, queue_size=1)
        self.pub_viz = rospy.Publisher('/coords_viz', Path, queue_size=1)
        rospy.loginfo('Coords Viz Node is Up!')
        rospy.spin()

    def coords_callback(self, data):
        # 0 - left_shoulder
        # 1 - right_shoulder
        # 2 - left_elbow
        # 3 - right_elbow
        # 4 - left_wrist
        # 5 - right_wrist
        coords = np_bridge.to_numpy_f64(data)
        h = Header(stamp=rospy.Time.now(), frame_id='map')
        poses = []
        poses.append(PoseStamped(header=h, pose=Pose(position=Point(coords[4, 0], coords[4, 1], coords[4, 2]), orientation=Quaternion(0, 0, 0, 1))))
        poses.append(PoseStamped(header=h, pose=Pose(position=Point(coords[2, 0], coords[2, 1], coords[2, 2]), orientation=Quaternion(0, 0, 0, 1))))
        poses.append(PoseStamped(header=h, pose=Pose(position=Point(coords[0, 0], coords[0, 1], coords[0, 2]), orientation=Quaternion(0, 0, 0, 1))))
        poses.append(PoseStamped(header=h, pose=Pose(position=Point(coords[1, 0], coords[1, 1], coords[1, 2]), orientation=Quaternion(0, 0, 0, 1))))
        poses.append(PoseStamped(header=h, pose=Pose(position=Point(coords[3, 0], coords[3, 1], coords[3, 2]), orientation=Quaternion(0, 0, 0, 1))))
        poses.append(PoseStamped(header=h, pose=Pose(position=Point(coords[5, 0], coords[5, 1], coords[5, 2]), orientation=Quaternion(0, 0, 0, 1))))
        self.pub_viz.publish(Path(header=h, poses=poses))


if __name__ == '__main__':
    CoordsViz()
