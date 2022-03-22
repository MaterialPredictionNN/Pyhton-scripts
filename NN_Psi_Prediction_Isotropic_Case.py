# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 15:39:00 2021

@author: cholv
"""

# =============================================================================================================
# Imports
# =============================================================================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from tensorflow import keras
from tensorflow.keras import layers
import numpy
import matplotlib.pyplot as plt
import time
import random
start_time = time.time()


# =============================================================================================================
# Raw data
# =============================================================================================================

def extract_data(path):
    crimefile = open(path, 'r')
    result = [line.split(',') for line in crimefile.readlines()]
    for i in range(len(result)):
        for j in range(len(result[0])):
            if result[i][j][0]=='[':
                result[i][j] = result[i][j][1:]
            if result[i][j][-1]==']':
                result[i][j] = result[i][j][:-1]
        result[i].pop(-1)
    
    for L in result :
        for i in range(len(L)):
            L[i] = float(L[i])
    crimefile.close()
    return result


# time data
T_List = extract_data(r'C:\Windows\System32\T_List.txt')

# x coordinates data
X_List = extract_data(r'C:\Windows\System32\X_List.txt')


# y coordinates data
Y_List = extract_data(r'C:\Windows\System32\Y_List.txt')
      

# x displacement data
U1_List = extract_data(r'C:\Windows\System32\U1_List.txt')


# y displacement data
U2_List = extract_data(r'C:\Windows\System32\U2_List.txt')


# external energy data
External_Energy_List = extract_data('C:\Windows\System32\External_Energy_List.txt')


# Dimension
x_dim = 60.0
y_dim = 10.0
z_dim = 5.0


# psi energy
v_tot = x_dim*y_dim*z_dim
Psi_List = []
for i in range(len(External_Energy_List)):
    Psi_List.append(External_Energy_List[i][0]/v_tot)


# =============================================================================================================
# Preparation for training data creation
# =============================================================================================================

x0 = X_List[0]
y0 = Y_List[0]

x0_unique = []
for i in range(len(x0)):
    if x0[i] not in x0_unique:
        x0_unique.append(x0[i])

x0_sort = sorted(x0_unique)

y0_unique = []
for i in range(len(y0)):
    if y0[i] not in y0_unique:
        y0_unique.append(y0[i])

y0_sort = sorted(y0_unique)

global Nodes_Indices
Nodes_Indices = []
indice = 0
for i in range(len(x0_sort)):
    Nodes_Indices_temp = []
    for j in range(len(y0_sort)):
        for k in range(len(x0)):
            if x0[k] == x0_sort[i] and y0[k] == y0_sort[j]:
                indice = k
        Nodes_Indices_temp.append(indice)
    Nodes_Indices.append(Nodes_Indices_temp)


# =============================================================================================================
# Gradient deformation matrix computation
# =============================================================================================================

def list2matrix(List):
    result = []
    for t in range(len(List)):
        result_t_temp = []
        for x in range(len(Nodes_Indices)):
            result_temp = []
            for y in range(len(Nodes_Indices[0])):
                result_temp.append(List[t][Nodes_Indices[x][y]])
            result_t_temp.append(result_temp)
        result.append(result_t_temp)
    return result    

U_Matrix = list2matrix(U1_List)
V_Matrix = list2matrix(U2_List)
X_Matrix = list2matrix(X_List)
Y_Matrix = list2matrix(Y_List)


F = []
for t in range(len(U_Matrix)):
    F_t_temp = []
    for x in range(1, len(U_Matrix[0])-1):
        F_temp = []
        for y in range(1, len(U_Matrix[0][0])-1):
            
            dux = (U_Matrix[t][x+1][y]-U_Matrix[t][x-1][y])/(X_Matrix[0][x+1][y]-X_Matrix[0][x-1][y])
            
            dvx = (V_Matrix[t][x+1][y]-V_Matrix[t][x-1][y])/(X_Matrix[0][x+1][y]-X_Matrix[0][x-1][y])
            
            duy = (U_Matrix[t][x][y+1]-U_Matrix[t][x][y-1])/(Y_Matrix[0][x][y+1]-Y_Matrix[0][x][y-1])
            
            dvy = (V_Matrix[t][x][y+1]-V_Matrix[t][x][y-1])/(Y_Matrix[0][x][y+1]-Y_Matrix[0][x][y-1])
            
            F11 = 1+dux
            F12 = duy
            F21 = dvx
            F22 = 1+dvy
            Fzz = 1/(F11*F22-F12*F21)
            F_temp.append([[F11, F12, 0], [F21, F22, 0], [0, 0, Fzz]])
        F_t_temp.append(F_temp)
    F.append(F_t_temp)


# =============================================================================================================
# Computation of the invariants (cf Bonnet & Wood - Nonlinear Continuum Mechanics for Finite Element Analysis)
# =============================================================================================================

Invariants = []
for t in range(len(F)):
    Invariants_t_temp = []
    for x in range(len(F[0])):
        Invariants_temp = []
        for y in range(len(F[0][0])):
            F_array_temp = numpy.asarray(F[t][x][y])
            C = numpy.dot(numpy.transpose(F_array_temp), F_array_temp)
            I_1 = numpy.trace(C)
            Invariants_temp.append(I_1)
        Invariants_t_temp.append(Invariants_temp)
    Invariants.append(Invariants_t_temp)


# =============================================================================================================
# Neural Network part
# =============================================================================================================

# Determination of training data for every time step (t=0, ..., t=tf)
Input_data = []
for t in range(len(Invariants)):
    c = 0
    Input_data_temp = 0
    for x in range(len(Invariants[0])):
        for y in range(len(Invariants[0][0])):
            c+=1
            Input_data_temp+=Invariants[t][x][y]-3
    Input_data.append(Input_data_temp/c)
    
Output_data = []
for t in range(len(Psi_List)):
    Output_data.append(Psi_List[t])


frac_train = 0.7
frac_test = 0.15
frac_validation = 0.15

n_tot = len(Input_data)
l_indice_temp = [i for i in range(n_tot)]
random.shuffle(l_indice_temp)

x_train = []
y_train = []

x_test = []
y_test = []

x_validation = []
y_validation = []

for i in range(n_tot):
    if i<frac_train*n_tot:
        x_train.append(Input_data[l_indice_temp[i]])
        y_train.append(Output_data[l_indice_temp[i]])
        
    if i>frac_train*n_tot and i<frac_train*n_tot+frac_test*n_tot:
        x_test.append(Input_data[l_indice_temp[i]])
        y_test.append(Output_data[l_indice_temp[i]])

    if i>frac_train*n_tot+frac_test*n_tot:
        x_validation.append(Input_data[l_indice_temp[i]])
        y_validation.append(Output_data[l_indice_temp[i]])
        
        
# Model ANN

model = keras.Sequential()

nb_neuron_1 = 1

Activation = keras.activations.linear

Epochs = 20
Batchsize = 1

model.add(keras.Input(shape=(1)))
model.add(layers.Dense(nb_neuron_1, activation = Activation))
model.add(layers.Dense(1))

inputs = keras.Input(shape=(1))
x = layers.Dense(nb_neuron_1, activation = Activation)(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs=inputs, outputs=outputs)


model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    loss = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error"),
    metrics = keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None),
)

print(model.summary())
history = model.fit(
		x_train,
		y_train,
		verbose = 2,
		batch_size = Batchsize,
		epochs = Epochs,
		shuffle = True,
		)
print('Data test')
model.evaluate(x_test, y_test, batch_size=2, verbose=2)


# =============================================================================================================
# Results plot
# =============================================================================================================

predictions = model.predict(x_validation)

l_indice_sort = []
x_validation_sort = sorted(x_validation)

for i in range(len(x_validation_sort)):
    j = 0
    while x_validation_sort[i] != x_validation[j]:
        j+=1
    l_indice_sort.append(j)

y_validation_plot = []
predictions_plot = []

for i in range(len(l_indice_sort)):
    indice = l_indice_sort[i]
    y_validation_plot.append(y_validation[indice])
    predictions_plot.append(predictions[indice])

for i in range(len(x_validation_sort)):
    x_validation_sort[i]+=3

plt.axis('equal')
plt.plot(x_validation_sort, y_validation_plot, 'r')
plt.plot(x_validation_sort, predictions_plot, 'b')
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.title('NN psi predictions for '+str(Epochs)+' epochs with '+str(nb_neuron_1)+' neurons on the first layer', fontsize=40)
plt.xlabel('First invariant', fontsize=40)
plt.ylabel('Psi', fontsize=40)
plt.legend(['Abaqus results', 'NN predictions'], loc='lower right', fontsize=35)
plt.show()


# =============================================================================================================
# Loss during training plot
# =============================================================================================================

plt.plot(history.history['loss'])
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.title('Loss error evolution during training', fontsize=40)
plt.xlabel('Epochs', fontsize=40)
plt.ylabel('Mean Squared Error', fontsize=40)
plt.show()
print(min(history.history['loss']))


# =============================================================================================================
# Get weights and biases
# =============================================================================================================

weights_1 = model.layers[1].get_weights()[0]
weights_1 = weights_1[0]
biases_1 = model.layers[1].get_weights()[1]

weights_2 = model.layers[2].get_weights()[0]
biases_2 = model.layers[2].get_weights()[1]

print('C10 predicted:', weights_1[0]*weights_2[0][0])

# =============================================================================================================
# End
# =============================================================================================================

print("--- %s seconds ---" % (time.time() - start_time))


# =============================================================================================================
# =============================================================================================================
# =============================================================================================================