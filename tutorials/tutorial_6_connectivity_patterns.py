"""
Connect populations of neurons with patterns more complicated than simple 'all-to-all' connectivity
William Nourse
January 12, 2022
"""
from sns_toolbox.design.connections import NonSpikingPatternConnection
from sns_toolbox.design.networks import Network
from sns_toolbox.design.neurons import NonSpikingNeuron

from sns_toolbox.simulate.backends import SNS_Numpy

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import sys

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IMAGE PREPROCESSING
"""
img = cv.imread('/home/will/sample_images/cameraman.png')   # load image file

if img is None:
    sys.exit('Could not find image')    # if image doesn't exist, exit

shape_original = img.shape  # dimensions of the original image
dim_long = max(shape_original[0],shape_original[1]) # longest dimension of the original image
dim_desired_max = 32    # constrain the longest dimension for easier processing
ratio = dim_desired_max/dim_long    # scaling ratio of original image
img_resized = cv.resize(img,None,fx=ratio,fy=ratio) # scale original image using ratio

img_color = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # transform the image from BGR to RGB
img_color_resized = cv.cvtColor(img_resized, cv.COLOR_BGR2RGB)  # resize the RGB image
img_gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)  # convert the resized image to grayscale [0-255]

shape = img_gray.shape  # dimensions of the resized grayscale image

img_flat = img_gray.flatten()   # flatten the image into 1 vector for neural processing
flat_size = len(img_flat)   # length of the flattened image vector

plt.ion()
plt.show()


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NETWORK CONSTRUCTION
"""
# General network
R = 20.0    # range of network activity (mV)
neuron_type = NonSpikingNeuron()    # generic neuron type
net = Network(name='Tutorial 6 Network')    # create an empty network

# Retina
net.add_population(neuron_type,shape,name='Retina') # add a 2d population the same size as the scaled image
net.add_input('Retina', size=flat_size,name='Image',    # add a vector input for the flattened scaled image
              linear=R/255) # scale all the intensities from 0-255 to 0-R
net.add_output('Retina',linear=255/R,name='Retina Output')  # add a vector output from the retina, scaled correctly

# Lamina
net.add_population(neuron_type,shape,name='Lamina')

del_e_ex = 160.0    # excitatory reversal potential
del_e_in = -80.0    # inhibitory reversal potential
k_ex = 1.0  # excitatory gain
k_in = -1.0/8.0 # inhibitory gain
g_max_ex = (k_ex*R)/(del_e_ex-k_ex*R)   # calculate excitatory conductance
g_max_in = (k_in*R)/(del_e_in-k_in*R)   # calculate inhibitory conductance

g_max_kernel = np.array([[g_max_in, g_max_in, g_max_in],    # kernel matrix of synaptic conductances
                         [g_max_in, g_max_ex, g_max_in],
                         [g_max_in, g_max_in, g_max_in]])
del_e_kernel = np.array([[del_e_in, del_e_in, del_e_in],    # kernel matrix of synaptic reversal potentials
                         [del_e_in, del_e_ex, del_e_in],
                         [del_e_in, del_e_in, del_e_in]])
connection_hpf = NonSpikingPatternConnection(g_max_kernel,del_e_kernel) # pattern connection (acts as high pass filter)
net.add_connection(connection_hpf,'Retina','Lamina',name='HPF') # connect the retina to the lamina
net.add_output('Lamina',linear=255/R,name='Lamina Output')  # add a vector output from the lamina

net.render_graph(view=True) # view the network diagram

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NETWORK COMPILATION
"""

dt = neuron_type.params['membrane_capacitance']/neuron_type.params['membrane_conductance']  # calculate the ideal dt
# dt = 0.5
t_max = 15  # run for 15 ms
steps = int(t_max/dt)   # number of steps to simulate

model = SNS_Numpy(net,dt=dt,debug=False) # compile using the numpy backend

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SIMULATION
"""
plt.figure()    # create a figure for live plotting the retina and lamina states
plt.subplot(1,2,1)
plt.title('Retina')
plt.axis('off')
plt.subplot(1,2,2)
plt.title('Lamina')
plt.axis('off')
plt.pause(0.5)

for i in range(steps):
    print('%i / %i steps'%(i+1,steps))
    out = model.forward(img_flat)   # run the network for one dt
    retina = out[:flat_size]    # separate the retina and lamina states
    lamina = out[flat_size:]
    retina_reshape = np.reshape(retina,shape)   # reshape to from flat to an image
    lamina_reshape = np.reshape(lamina,shape)
    plt.subplot(1,2,1)  # plot the current state
    plt.imshow(retina_reshape,cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(lamina_reshape, cmap='gray')
    plt.pause(0.001)

plt.ioff()

# plt.figure()
# plt.imshow(img_color)
# plt.axis('off')
# plt.title('Color Original')
# # plt.pause(0.5)
#
# plt.figure()
# plt.imshow(img_color_resized)
# plt.axis('off')
# plt.title('Scaled Down')
# # plt.pause(0.5)
#
# plt.figure()
# plt.imshow(img_gray, cmap='gray')
# plt.axis('off')
# plt.title('Grayscale')
# # plt.pause(0.5)

print('DONE')
plt.show()
