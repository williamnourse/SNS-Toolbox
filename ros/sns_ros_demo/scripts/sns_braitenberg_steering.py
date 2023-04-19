#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from rospy.numpy_msg import numpy_msg
import torch
from sns_toolbox.networks import Network
from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.connections import NonSpikingMatrixConnection, NonSpikingSynapse

global scan_angles
scan_angles = np.zeros(720)

def gen_net(num_scan):
    net = Network()
    neuron = NonSpikingNeuron(membrane_capacitance=5.0, membrane_conductance=1.0,resting_potential=0.0)
    net.add_population(neuron,[num_scan],name='Scan')
    net.add_input('Scan', size=num_scan)

    net.add_population(neuron,[2], name='Heading')
    net.add_output('Heading')

    net.add_neuron(neuron,name='Velocity')
    net.add_output('Velocity')

    g_max = 0.005
    g_matrix = np.zeros([2,num_scan])
    rev_matrix = np.zeros_like(g_matrix)+5.0
    e_lo_matrix = np.zeros_like(g_matrix)
    e_hi_matrix = np.ones_like(g_matrix)
    g_matrix[0,int(num_scan/2):] = g_max
    g_matrix[1,:int(num_scan/2)] = g_max
    conn_heading = NonSpikingMatrixConnection(g_matrix,rev_matrix,e_lo_matrix,e_hi_matrix)

    conn_velocity = NonSpikingSynapse(max_conductance=1.0, reversal_potential=5.0,e_lo=0.0,e_hi=1.0)

    net.add_connection(conn_heading,'Scan','Heading')
    net.add_connection(conn_velocity,'Scan','Velocity')

    model = net.compile(dt=1, backend='torch', device='cuda')
    return model

def condition_input(scan, min_val, max_val):
    inp = torch.Tensor(scan.copy()).to('cuda')
    inp = ((1/inp)-min_val)/(max_val-min_val)
    return inp

def condition_output(out_data):
    out_pos = torch.clamp(out_data,min=0.0,max=1.0)
    heading = out_pos[1]-out_pos[0]
    heading_scaled = 10*heading/(2*np.pi)
    speed = out_pos[2]*2.0
    return heading_scaled, speed*2

def run_node():
    rospy.loginfo('Creating Velocity Publisher')
    vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    neurons_pub = rospy.Publisher('/neurons', Float32MultiArray, queue_size=10)
    # rospy.sleep(30)
    #input('Gazebo Ready? Press Enter to Continue.')
    # rospy.init_node('sns_controller', anonymous=True)
    range_min = 0.1
    min_inv = 1/range_min
    range_max = 30.0
    max_inv = 1/range_max
    num_angles = 720
    vel_msg = Twist()
    neurons_msg = Float32MultiArray()
    max_speed = 2.0
    max_angle = 2 * np.pi

    vel_msg.linear.x = max_speed
    vel_msg.linear.y = 0.0
    vel_msg.linear.z = 0.0
    vel_msg.angular.x = 0.0
    vel_msg.angular.y = 0.0
    vel_msg.angular.z = 0.0

    rospy.loginfo('Building Network')
    model = gen_net(num_angles)
    rospy.loginfo('Network Built')
    rospy.loginfo('Begin Guidance')

    # rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        inp = condition_input(scan_angles,max_inv,min_inv)
        # for i in range(10):
        data = model(inp)
        # head_ccw_pub.publish(data[0].to('cpu'))
        # head_cw_pub.publish(data[1].to('cpu'))
        # vel_neuron_pub.publish(data[2].to('cpu'))

        data_cpu = data.to('cpu')
        neurons_msg.data = data_cpu
        neurons_pub.publish(neurons_msg)
        heading, speed = condition_output(data_cpu)
        vel_msg.angular.z = heading
        vel_msg.linear.x = speed
        vel_pub.publish(vel_msg)
        rospy.loginfo('Linear Velocity: %.2f, Angular Velocity: %.2f'%(speed,heading))
        # rospy.loginfo(scan_angles)

def scan_callback(data):
    # print(len(data.ranges))
    global scan_angles
    scan_angles = data.ranges


if __name__ == '__main__':
    try:
        rospy.loginfo('Initializing Node')
        rospy.init_node('sns_controller', anonymous=True)
        rospy.loginfo('Creating Scan Subscriber')
        rospy.Subscriber('/front/scan', numpy_msg(LaserScan),scan_callback)
        run_node()
    except rospy.ROSInterruptException: pass
