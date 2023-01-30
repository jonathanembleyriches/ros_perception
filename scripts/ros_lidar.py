
import rospy
import sensor_msgs.point_cloud2 as pc2
import rospy, math
import numpy as np
from sensor_msgs.msg import LaserScan
from laser_geometry import LaserProjection


class Lidar():
    def __init__(self, scan_topic="/robot_0/base_scan"):
        rospy.init_node("LiDAr_processor")
        self.scan_sub = rospy.Subscriber(scan_topic, LaserScan, self.on_scan)
        self.laser_projector = LaserProjection()
        rospy.loginfo("Init complete. Starting to listen")

    def on_scan(self, scan):
        rospy.loginfo("Got scan, projecting")
        cloud = self.laser_projector.projectLaser(scan)
        gen = pc2.read_points(cloud, skip_nans=True, field_names=("x", "y", "z"))
        self.xyz_generator = gen
        print(f"Laser scan: ", gen)

if __name__ == '__main__':
    try:
        lidar = Lidar()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
