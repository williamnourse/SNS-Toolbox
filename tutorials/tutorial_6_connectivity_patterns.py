"""
Connect populations of neurons with patterns more complicated than simple 'all-to-all' connectivity
William Nourse
January 12, 2022
"""
import cv2

from sns_toolbox.design.connections import NonSpikingPatternConnection
from sns_toolbox.design.networks import Network
from sns_toolbox.design.neurons import NonSpikingNeuron

from sns_toolbox.simulate.backends import SNS_Numpy

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import sys

img = cv.imread('/home/will/sample_images/starry_night.jpg')

if img is None:
    sys.exit('Could not find image')

# cv.imshow('Display window', img)
# k = cv.waitKey(0)
shape_original = img.shape
dim_long = max(shape_original[0],shape_original[1])
dim_desired_max = 64
ratio = dim_desired_max/dim_long
img_resized = cv2.resize(img,None,fx=ratio,fy=ratio)

img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

shape = img_gray.shape
# print(shape)

img_flat = img_gray.flatten()
max_intensity = max(img_flat)
print(max_intensity)

R = 20.0
neuron_type = NonSpikingNeuron()
net = Network(name='Tutorial 6 Network')
net.add_population(neuron_type,shape,name='Retina')
net.add_input('Retina', size=len(img_flat),name='Image',linear=R/max_intensity)
net.add_output('Retina',linear=255/R,name='Output')

net.render_graph(view=True)

dt = 5
t_max = 100
steps = int(t_max/dt)

model = SNS_Numpy(net,dt=dt)

plt.figure()
plt.ion()
plt.show()
for i in range(steps):
    print('%i / %i'%(i,steps))
    out = model.forward(img_flat)
    out_reshape = np.reshape(out,shape)
    plt.imshow(out_reshape,cmap='gray')
    plt.pause(0.001)

plt.ioff()
plt.figure()
plt.imshow(img_color)
plt.title('Color Original')

plt.figure()
plt.imshow(img_resized)
plt.title('Scaled Down')

plt.figure()
plt.imshow(img_gray, cmap='gray')
plt.title('Grayscale')

plt.show()
