#!/usr/bin/env python3

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose

def callback(data):
    global pub
    index = data.name.index("jackal")
    pose = data.pose[index]
    pub.publish(pose)


if __name__ == '__main__':
    rospy.loginfo('Initializing Node')
    rospy.init_node('model_states_listener', anonymous=True)
    rospy.loginfo('Creating Model Subscriber')
    pub = rospy.Publisher('jackal_pose', Pose, queue_size=10)
    rospy.Subscriber('/gazebo/model_states', ModelStates, callback)
    rospy.loginfo('Running')
    rospy.spin()
