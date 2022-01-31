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

# x coordinates data
crimefile = open('C:\Windows\System32\X_List.txt', 'r')
X_List = [line.split(',') for line in crimefile.readlines()]
for i in range(len(X_List)):
    for j in range(len(X_List[0])):
        if X_List[i][j][0]=='[':
            X_List[i][j] = X_List[i][j][1:]
        if X_List[i][j][-1]==']':
            X_List[i][j] = X_List[i][j][:-1]
    X_List[i].pop(-1)

for L in X_List :
    for i in range(len(L)):
        L[i] = float(L[i])
crimefile.close()


# y coordinates data
crimefile = open('C:\Windows\System32\Y_List.txt', 'r')
Y_List = [line.split(',') for line in crimefile.readlines()]
for i in range(len(Y_List)):
    for j in range(len(Y_List[0])):
        if Y_List[i][j][0]=='[':
            Y_List[i][j] = Y_List[i][j][1:]
        if Y_List[i][j][-1]==']':
            Y_List[i][j] = Y_List[i][j][:-1]
    Y_List[i].pop(-1)
        
for L in Y_List :
    for i in range(len(L)):
        L[i] = float(L[i])
crimefile.close()
      

# x displacement data
crimefile = open(r'C:\Windows\System32\U1_List.txt', 'r')
U1_List = [line.split(',') for line in crimefile.readlines()]
for i in range(len(U1_List)):
    for j in range(len(U1_List[0])):
        if U1_List[i][j][0]=='[':
            U1_List[i][j] = U1_List[i][j][1:]
        if U1_List[i][j][-1]==']':
            U1_List[i][j] = U1_List[i][j][:-1]
    U1_List[i].pop(-1)
        
for L in U1_List :
    for i in range(len(L)):
        L[i] = float(L[i])
crimefile.close()


# y displacement data
crimefile = open(r'C:\Windows\System32\U2_List.txt', 'r')
U2_List = [line.split(',') for line in crimefile.readlines()]
for i in range(len(U2_List)):
    for j in range(len(U2_List[0])):
        if U2_List[i][j][0]=='[':
            U2_List[i][j] = U2_List[i][j][1:]
        if U2_List[i][j][-1]==']':
            U2_List[i][j] = U2_List[i][j][:-1]

    U2_List[i].pop(-1)
        
for L in U2_List :
    for i in range(len(L)):
        L[i] = float(L[i])
crimefile.close()


# stress data
crimefile = open('C:\Windows\System32\Stress_List.txt', 'r')
Stress_List = [line.split(',') for line in crimefile.readlines()]
for i in range(len(Stress_List)):
    for j in range(len(Stress_List[0])):
        if Stress_List[i][j][0]=='[':
            Stress_List[i][j] = Stress_List[i][j][1:]
        if Stress_List[i][j][-1]==']':
            Stress_List[i][j] = Stress_List[i][j][:-1]
    Stress_List[i].pop(-1)
        
for L in Stress_List :
    for i in range(len(L)):
        L[i] = float(L[i])
crimefile.close()


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

def remove_border(Matrix):
    result = []
    temp = Matrix[1:len(Matrix)-1]
    for i in range(len(temp)):
        result.append(temp[i][1:len(temp[i])-1])
    return result

def diff_matrix(m1, m2, idiff_1, idiff_2):
    result = []
    for t in range(len(m1)):
        temp_1 = numpy.gradient(m1[t])[idiff_1]
        diff_temp_1 = numpy.array(remove_border(temp_1))
        temp_2 = numpy.gradient(m2[t])[idiff_2]
        diff_temp_2 = numpy.array(remove_border(temp_2))
        result.append(diff_temp_1/diff_temp_2)
    return result
    

    
Stress_Matrix = list2matrix(Stress_List)
U_Matrix = list2matrix(U1_List)
V_Matrix = list2matrix(U2_List)
X_Matrix = list2matrix(X_List)
Y_Matrix = list2matrix(Y_List)


du_dx = diff_matrix(U_Matrix, X_Matrix, 0, 0)
du_dy = diff_matrix(U_Matrix, Y_Matrix, 1, 1)

dv_dx = diff_matrix(V_Matrix, X_Matrix, 0, 0)
dv_dy = diff_matrix(V_Matrix, Y_Matrix, 1, 1)

x_coord = []
y_coord = []
stress_field = []
for t in range(len(X_Matrix)):
    x_coord.append(remove_border(X_Matrix[t]))
    y_coord.append(remove_border(Y_Matrix[t]))
    stress_field.append(remove_border(Stress_Matrix[t]))


# Validation plots

# y_plot = []
# x_plot = []
# t = 10
# for x in range(len(x_coord[t])):
#     for y in range(len(x_coord[t][0])):
#         x_plot.append(y_coord[t][x][y])
#         y_plot.append(dv_dy[t][x][y])

# plt.plot(x_plot, y_plot)
# plt.axis([2, 10, -0.1, 0.1])
# plt.show()


F = []
for t in range(len(du_dx)):
    F_t_temp = []
    for x in range(len(du_dx[0])):
        F_temp = []
        for y in range(len(du_dx[0][0])):
            F11 = 1+du_dx[t][x][y]
            F12 = du_dy[t][x][y]
            F21 = dv_dx[t][x][y]
            F22 = 1+dv_dy[t][x][y]
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
            I_C = numpy.matrix.trace(C)
            II_C = numpy.matrix.trace(numpy.dot(C, C))
            III_C = numpy.linalg.det(C)
            Invariants_temp.append([I_C, II_C, III_C])
        Invariants_t_temp.append(Invariants_temp)
    Invariants.append(Invariants_t_temp)


# =============================================================================================================
# Neural Network part
# =============================================================================================================

# Determination of training data for every time step (t=0, ..., t=11)

Input_data = []
for t in range(len(Invariants)):
    for x in range(len(Invariants[0])):
        for y in range(len(Invariants[0][0])):
            Input_data.append(Invariants[t][x][y][0])
            
Output_data = []
for t in range(len(stress_field)):
    for x in range(len(stress_field[0])):
        for y in range(len(stress_field[0][0])):
            Output_data.append(stress_field[t][x][y])
            
frac_train = 0.7
frac_test = 0.1
frac_validation = 0.2

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


# Model NN

model = keras.Sequential()

nb_neuron_1 = 6
#nb_neuron_2 = 8

Activation = keras.activations.tanh

Epochs = 3000
Batchsize = 4

model.add(keras.Input(shape=(1)))
model.add(layers.Dense(nb_neuron_1, activation = Activation))
#model.add(layers.Dense(nb_neuron_2, activation = Activation))
model.add(layers.Dense(1))

inputs = keras.Input(shape=(1))
x = layers.Dense(nb_neuron_1, activation = Activation)(inputs)
#x = layers.Dense(nb_neuron_2, activation = Activation)(x)
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
model.evaluate(x_test, y_test, batch_size=4, verbose=2)

s = 0
c = 0
for i in range(len(y_test)):
    c+=1
    s+=y_test[i]**2

print('mean squared value of test:', s/c)


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
    

plt.plot(x_validation_sort, y_validation_plot, 'r')
plt.plot(x_validation_sort, predictions_plot, 'b')
#plt.title('NN stress predictions for '+str(Epochs)+' epochs with '+str(nb_neuron_1)+' neurons on the first layer and '+str(nb_neuron_2)+' on the second layer', fontsize=20)
plt.title('NN stress predictions for '+str(Epochs)+' epochs with '+str(nb_neuron_1)+' neurons on the first layer', fontsize=20)
plt.xlabel('First invariant', fontsize=15)
plt.ylabel('Stress', fontsize=15)
plt.legend(['Abaqus results', 'NN predictions'], loc='lower right', fontsize=18)
plt.show()

# plt.plot(history.history['loss'])
# plt.show()
# print(min(history.history['loss']))

print("--- %s seconds ---" % (time.time() - start_time))