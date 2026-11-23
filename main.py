# one input layer, two hidden layers and one output layer

# input = [1.0,2.0,3.0]

# weight1 = [0.2, 0.8, -0.5]
# weight2 = [0.5, 0.91, 0.26]
# weight3 = [-0.26, -0.27, 0.17]
# weight4 = [0.1, -0.14, 0.5]

# bias1 = 2.0
# bias2 = 3.0
# bias3 = 0.5
# bias4 = 1.0

# neuron1 = input[0] * weight1[0] + input[1] * weight1[1] + input[2] * weight1[2] + bias1
# neuron2 = input[0] * weight2[0] + input[1] * weight2[1] + input[2]* weight2[2] + bias2
# neuron3 = input[0] * weight3[0] + input[1] * weight3[1] + input[2]* weight3[2] + bias3
# neuron4 = input[0] * weight4[0] + input[1] * weight4[1] + input[2]* weight4[2] + bias4
# output = [neuron1, neuron2, neuron3, neuron4]


# print(output)

########################################################################################################

# inputs = [1.0,2.0,3.0]
# weights = [[0.2, 0.8, -0.5],[0.5, 0.91, 0.26], [-0.26, -0.27, 0.17], [0.1, -0.14, 0.5]
    
# ]
# biases = [2.0, 3.0, 0.5, 1.0]

# outputs = []

# for neuron_weights, bias in zip(weights, biases):
#     neuron_output = sum(
#         neuron_weight * input for input, neuron_weight in zip(inputs, neuron_weights)
        
#     )
#     neuron_output +=bias
#     outputs.append(neuron_output)
    
    
# print([round(output, 2)for output in outputs])

###################################################################################################################
import numpy as np

inputs = [1.0,2.0,3.0]
weights = [[0.2, 0.8, -0.5],[0.5, 0.91, 0.26], [-0.26, -0.27, 0.17], [0.1, -0.14, 0.5]
    
]
biases = [2.0, 3.0, 0.5, 1.0]

output = np.dot(weights, inputs)
print(output + biases)



    