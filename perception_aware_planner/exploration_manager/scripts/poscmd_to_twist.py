#!/usr/bin/env python3
# coding: utf-8
"""
poscmd_to_twist.py
订阅:  /planning/pos_cmd (quadrotor_msgs/PositionCommand)
发布:  /kingfisher/agiros_pilot/velocity_command (geometry_msgs/TwistStamped)

映射关系（默认世界系）:
  twist.linear  <- PositionCommand.velocity (x,y,z)
  twist.angular <- (0, 0, PositionCommand.yaw_dot)

可选参数:
  ~vel_in_body   (bool, default: False) 若为 True，将 velocity 由世界系旋到机体系（用 yaw）
  ~output_frame  (string, default: 自动选择；机体系时为 base_link，否则沿用输入 header.frame_id 或 "world")
  ~queue_size    (int, default: 50)
"""

import math
import rospy
import numpy as np
from geometry_msgs.msg import TwistStamped, Vector3
from std_msgs.msg import Header
from quadrotor_msgs.msg import PositionCommand
from nav_msgs.msg import Odometry

class PosCmdToTwist:
    def __init__(self):
        self.vel_in_body = rospy.get_param("~vel_in_body", False)
        self.output_frame_param = rospy.get_param("~output_frame", "")
        qsize = int(rospy.get_param("~queue_size", 50))

        # 控制增益
        self.kp = rospy.get_param("~kp", 0.1)
        self.kd = rospy.get_param("~kd", 0.1)

        self.pub = rospy.Publisher(
            "/kingfisher/agiros_pilot/velocity_command", TwistStamped, queue_size=qsize
        )
        self.sub_cmd = rospy.Subscriber(
            "/planning/pos_cmd", PositionCommand, self.cb, queue_size=qsize
        )
        self.sub_odom = rospy.Subscriber(
            "/drone/odom", Odometry, self.odom_cb, queue_size=1
        )

        self.latest_odom = None

        rospy.loginfo("[poscmd_to_twist] vel_in_body=%s, output_frame='%s' kp=%.2f kd=%.2f",
                      str(self.vel_in_body), self.output_frame_param if self.output_frame_param else "(auto)",
                      self.kp, self.kd)

    def odom_cb(self, msg: Odometry):
        self.latest_odom = msg

    @staticmethod
    def world_vel_to_body(vx, vy, vz, yaw):
        """把世界系速度旋到机体系（绕 z 轴）"""
        c = math.cos(yaw)
        s = math.sin(yaw)
        bx =  c * vx + s * vy
        by = -s * vx + c * vy
        bz =  vz
        return bx, by, bz

    def cb(self, msg: PositionCommand):
        vx, vy, vz = float(msg.velocity.x), float(msg.velocity.y), float(msg.velocity.z)
        yaw, yaw_dot = float(msg.yaw), float(msg.yaw_dot)

        # ========== PD 修正 ==========
        if self.latest_odom is not None:
            # 位置误差
            ex = msg.position.x - self.latest_odom.pose.pose.position.x
            ey = msg.position.y - self.latest_odom.pose.pose.position.y
            ez = msg.position.z - self.latest_odom.pose.pose.position.z

            # 当前实际速度
            vx_real = self.latest_odom.twist.twist.linear.x
            vy_real = self.latest_odom.twist.twist.linear.y
            vz_real = self.latest_odom.twist.twist.linear.z

            # 反馈项
            vx += self.kp * ex - self.kd * vx_real
            vy += self.kp * ey - self.kd * vy_real
            vz += self.kp * ez - self.kd * vz_real

        # 如果要在机体系输出
        if self.vel_in_body:
            vx, vy, vz = self.world_vel_to_body(vx, vy, vz, yaw)

        out = TwistStamped()
        out.header = Header(stamp=msg.header.stamp)
        if self.output_frame_param:
            out.header.frame_id = self.output_frame_param
        else:
            out.header.frame_id = "base_link" if self.vel_in_body else (msg.header.frame_id or "world")

        out.twist.linear = Vector3(x=vx, y=vy, z=vz)
        out.twist.angular = Vector3(x=0.0, y=0.0, z=yaw_dot)

        self.pub.publish(out)

        rospy.logdebug_throttle(1.0,
            "[poscmd_to_twist] v=(%.2f,%.2f,%.2f) yaw_dot=%.2f frame=%s",
            vx, vy, vz, yaw_dot, out.header.frame_id
        )

def main():
    rospy.init_node("poscmd_to_twist", anonymous=False)
    PosCmdToTwist()
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
