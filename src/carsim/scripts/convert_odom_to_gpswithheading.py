#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import PoseStamped, Pose, Quaternion
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from sensor_driver_msgs.msg import GpswithHeading  # 假设消息定义在一个名为 `custom_msgs` 的包中

# 地球半径（单位：米）
EARTH_RADIUS = 6371000

# 初始 GPS 经纬度（参考点，北京）
INITIAL_LATITUDE = 39.9042
INITIAL_LONGITUDE = 116.4074

def odom_to_gps(x, y, z):
    """
    将里程计 (Odom) 相对位置 (x, y, z) 转换为 GPS 经纬度。
    
    :param x: Odom 的相对 X 位置（单位：米）
    :param y: Odom 的相对 Y 位置（单位：米）
    :param z: Odom 的相对 Z 位置（单位：米），用作高度（海拔）
    :return: GPS 经纬度和海拔
    """
    # 计算纬度变化量（ΔLatitude）
    delta_latitude = (y / EARTH_RADIUS) * (180 / math.pi)
    
    # 计算经度变化量（ΔLongitude）
    delta_longitude = (x / (EARTH_RADIUS * math.cos(math.radians(INITIAL_LATITUDE)))) * (180 / math.pi)
    
    # 转换为新的 GPS 经纬度
    latitude = INITIAL_LATITUDE + delta_latitude
    longitude = INITIAL_LONGITUDE + delta_longitude
    altitude = z  # 直接使用 Odom 的 z 作为海拔

    return latitude, longitude, altitude

def quaternion_to_euler(q):
    """
    Converts quaternion to Euler angles (roll, pitch, yaw).
    """
    x, y, z, w = q.x, q.y, q.z, q.w
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = math.asin(sinp) if abs(sinp) <= 1 else math.copysign(math.pi / 2, sinp)

    # Yaw (z-axis rotation) -> Heading
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def pose_callback(pose_msg):
    """
    Callback to handle incoming center_pose messages.
    """

    x = pose_msg.pose.position.x
    y = pose_msg.pose.position.y
    z = pose_msg.pose.position.z
    latitude, longitude, altitude = odom_to_gps(x, y, z)

    # 创建并发布 GpswithHeading 消息
    gps_msg = GpswithHeading()
    gps_msg.header.stamp = rospy.Time.now()
    gps_msg.gps.latitude = latitude
    gps_msg.gps.longitude = longitude
    gps_msg.gps.altitude = altitude

    pitch, roll, yaw = quaternion_to_euler(pose_msg.pose.orientation)
    gps_msg.heading = math.degrees(yaw)  # Convert to degrees
    gps_msg.roll = math.degrees(roll)  # Convert to degrees
    gps_msg.pitch = math.degrees(pitch)  # Convert to degrees

    # Add a dummy mode (could be refined based on specific requirements)
    gps_msg.mode = 0  # Example mode value

    # Publish the converted message
    gps_pub.publish(gps_msg)

    # 创建并发布 Odometry 消息
    odom_msg = Odometry()
    odom_msg.header = gps_msg.header  # 使用相同的时间戳和帧 ID
    odom_msg.pose.pose = Pose(pose_msg.pose.position, pose_msg.pose.orientation)  # 直接从 PoseStamped 赋值

    # 设置里程计的其他信息
    odom_msg.child_frame_id = "base_link"  # 子坐标系 ID（可调整）
    odom_msg.twist.twist.linear.x = 0.0  # 如果有速度信息，可以填入实际值
    odom_msg.twist.twist.linear.y = 0.0
    odom_msg.twist.twist.linear.z = 0.0
    odom_msg.twist.twist.angular.x = 0.0
    odom_msg.twist.twist.angular.y = 0.0
    odom_msg.twist.twist.angular.z = 0.0

    # 发布 Odometry 消息
    odom_pub.publish(odom_msg)


def main():
    rospy.init_node('odom_to_gpswithheading')
    rospy.Subscriber('/racebot/true_state/center_pose', PoseStamped, pose_callback)
    global gps_pub, odom_pub
    gps_pub = rospy.Publisher('/liorf/gpsdata', GpswithHeading, queue_size=10)
    odom_pub = rospy.Publisher('/liorf/mapping/odometry', Odometry, queue_size=10)
    rospy.spin()

if __name__ == '__main__':
    main()
