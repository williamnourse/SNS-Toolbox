"""
Connect populations of neurons with patterns more complicated than simple 'all-to-all' connectivity
William Nourse
January 12, 2022
"""

from sns_toolbox.design.connections import PatternConnection

vector_kernel = [-0.25,-0.25,1.0,-0.25,-0.25]
matrix_kernel = [[-0.125,-0.125,-0.125],
                 [-0.125,1.0,-0.125],
                 [-0.125,-0.125,-0.125]]
matrix_kernel_w_zeros = [[0.0,-0.25,0.0],
                         [-0.25,1.0,-0.25],
                         [0.0,-0.25,0.0]]

vector_connection = PatternConnection(vector_kernel)
matrix_connection = PatternConnection(matrix_kernel)
matrix_w_zeros_connection = PatternConnection(matrix_kernel_w_zeros)

print('Done')
