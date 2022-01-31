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
import math
start_time = time.time()

# =============================================================================================================
# Raw data
# =============================================================================================================

def extract_list_from_text(file_directory_name):
    crimefile = open(file_directory_name, 'r')
    list_result = [line.split(',') for line in crimefile.readlines()]
    for i in range(len(list_result)):
        for j in range(len(list_result[0])):
            if list_result[i][j][0]=='[':
                list_result[i][j] = list_result[i][j][1:]
            if list_result[i][j][-1]==']':
                list_result[i][j] = list_result[i][j][:-1]
        list_result[i].pop(-1)

    for L in list_result :
        for i in range(len(L)):
            L[i] = float(L[i])
    crimefile.close()
    return(list_result)



# x coordinates data
X_List_1 = extract_list_from_text(r'C:\Windows\System32\X_List_1.txt')
X_List_2 = extract_list_from_text(r'C:\Windows\System32\X_List_2.txt')
X_List_3 = extract_list_from_text(r'C:\Windows\System32\X_List_3.txt')


# y coordinates data
Y_List_1 = extract_list_from_text('C:\Windows\System32\Y_List_1.txt')
Y_List_2 = extract_list_from_text('C:\Windows\System32\Y_List_2.txt')
Y_List_3 = extract_list_from_text('C:\Windows\System32\Y_List_3.txt')
      

# x displacement data
U1_List_1 = extract_list_from_text(r'C:\Windows\System32\U1_List_1.txt')
U1_List_2 = extract_list_from_text(r'C:\Windows\System32\U1_List_2.txt')
U1_List_3 = extract_list_from_text(r'C:\Windows\System32\U1_List_3.txt')


# y displacement data
U2_List_1 = extract_list_from_text(r'C:\Windows\System32\U2_List_1.txt')
U2_List_2 = extract_list_from_text(r'C:\Windows\System32\U2_List_2.txt')
U2_List_3 = extract_list_from_text(r'C:\Windows\System32\U2_List_3.txt')


# external energy data
External_Energy_List_1 = extract_list_from_text('C:\Windows\System32\External_Energy_List_1.txt')
External_Energy_List_2 = extract_list_from_text('C:\Windows\System32\External_Energy_List_2.txt')
External_Energy_List_3 = extract_list_from_text('C:\Windows\System32\External_Energy_List_3.txt')


# Dimension

x_dim = 20.0
y_dim = 20.0
z_dim = 2.0


# Fibers orientation

gamma = 49.98
A = numpy.asarray([math.cos(gamma), math.sin(gamma), 0.0])
B = numpy.asarray([math.cos(gamma), -math.sin(gamma), 0.0])

N1 = [[A[0]**2, A[0]*A[1], 0.0], [A[0]*A[1], A[1]**2, 0.0], [0.0, 0.0, 0.0]]
N2 = [[B[0]**2, B[0]*B[1], 0.0], [B[0]*B[1], B[1]**2, 0.0], [0.0, 0.0, 0.0]]

# psi energy
v_tot = x_dim*y_dim*z_dim
Psi_List_1 = []
Psi_List_2 = []
Psi_List_3 = []
for i in range(len(External_Energy_List_1)):
#    if not (i==1 or i==2 or i==3 or i==4 or i==5 or i==7):
    Psi_List_1.append(External_Energy_List_1[i][0]/v_tot)
    Psi_List_2.append(External_Energy_List_2[i][0]/v_tot)
    Psi_List_3.append(External_Energy_List_3[i][0]/v_tot)


# =============================================================================================================
# Preparation for training data creation
# =============================================================================================================

x0 = X_List_1[0]
y0 = Y_List_1[0]

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
  
# Experiment 1  

U_Matrix_1 = list2matrix(U1_List_1)
V_Matrix_1 = list2matrix(U2_List_1)
X_Matrix_1 = list2matrix(X_List_1)
Y_Matrix_1 = list2matrix(Y_List_1)

du_dx_1 = diff_matrix(U_Matrix_1, X_Matrix_1, 0, 0)
du_dy_1 = diff_matrix(U_Matrix_1, Y_Matrix_1, 1, 1)
dv_dx_1 = diff_matrix(V_Matrix_1, X_Matrix_1, 0, 0)
dv_dy_1 = diff_matrix(V_Matrix_1, Y_Matrix_1, 1, 1)

F_1 = []
for t in range(len(du_dx_1)):
    F_1_t_temp = []
    for x in range(len(du_dx_1[0])):
        F_1_temp = []
        for y in range(len(du_dx_1[0][0])):
            F11 = 1+du_dx_1[t][x][y]
            F12 = du_dy_1[t][x][y]
            F21 = dv_dx_1[t][x][y]
            F22 = 1+dv_dy_1[t][x][y]
            Fzz = 1/(F11*F22-F12*F21)
            F_1_temp.append([[F11, F12, 0], [F21, F22, 0], [0, 0, Fzz]])
        F_1_t_temp.append(F_1_temp)
    F_1.append(F_1_t_temp)


# Experiment 2

U_Matrix_2 = list2matrix(U1_List_2)
V_Matrix_2 = list2matrix(U2_List_2)
X_Matrix_2 = list2matrix(X_List_2)
Y_Matrix_2 = list2matrix(Y_List_2)

du_dx_2 = diff_matrix(U_Matrix_2, X_Matrix_2, 0, 0)
du_dy_2 = diff_matrix(U_Matrix_2, Y_Matrix_2, 1, 1)
dv_dx_2 = diff_matrix(V_Matrix_2, X_Matrix_2, 0, 0)
dv_dy_2 = diff_matrix(V_Matrix_2, Y_Matrix_2, 1, 1)

F_2 = []
for t in range(len(du_dx_2)):
    F_2_t_temp = []
    for x in range(len(du_dx_2[0])):
        F_2_temp = []
        for y in range(len(du_dx_2[0][0])):
            F11 = 1+du_dx_2[t][x][y]
            F12 = du_dy_2[t][x][y]
            F21 = dv_dx_2[t][x][y]
            F22 = 1+dv_dy_2[t][x][y]
            Fzz = 1/(F11*F22-F12*F21)
            F_2_temp.append([[F11, F12, 0], [F21, F22, 0], [0, 0, Fzz]])
        F_2_t_temp.append(F_2_temp)
    F_2.append(F_2_t_temp)

# Experiment 3  

U_Matrix_3 = list2matrix(U1_List_3)
V_Matrix_3 = list2matrix(U2_List_3)
X_Matrix_3 = list2matrix(X_List_3)
Y_Matrix_3 = list2matrix(Y_List_3)

du_dx_3 = diff_matrix(U_Matrix_3, X_Matrix_3, 0, 0)
du_dy_3 = diff_matrix(U_Matrix_3, Y_Matrix_3, 1, 1)
dv_dx_3 = diff_matrix(V_Matrix_3, X_Matrix_3, 0, 0)
dv_dy_3 = diff_matrix(V_Matrix_3, Y_Matrix_3, 1, 1)

F_3 = []
for t in range(len(du_dx_3)):
    F_3_t_temp = []
    for x in range(len(du_dx_3[0])):
        F_3_temp = []
        for y in range(len(du_dx_3[0][0])):
            F11 = 1+du_dx_3[t][x][y]
            F12 = du_dy_3[t][x][y]
            F21 = dv_dx_3[t][x][y]
            F22 = 1+dv_dy_3[t][x][y]
            Fzz = 1/(F11*F22-F12*F21)
            F_3_temp.append([[F11, F12, 0], [F21, F22, 0], [0, 0, Fzz]])
        F_3_t_temp.append(F_3_temp)
    F_3.append(F_3_t_temp)


# =============================================================================================================
# Computation of the invariants (cf Bonnet & Wood - Nonlinear Continuum Mechanics for Finite Element Analysis)
# =============================================================================================================

# Experiment 1

Invariants_1 = []
for t in range(len(F_1)):
    Invariants_1_t_temp = []
    for x in range(len(F_1[0])):
        Invariants_1_temp = []
        for y in range(len(F_1[0][0])):
            F_1_array_temp = numpy.asarray(F_1[t][x][y])
            C = numpy.dot(numpy.transpose(F_1_array_temp), F_1_array_temp)
            I_1 = numpy.matrix.trace(C)
            I_4 = numpy.matrix.trace(numpy.dot(C, N1))
            I_6 = numpy.matrix.trace(numpy.dot(C, N2))
            Invariants_1_temp.append([I_1, I_4, I_6])
        Invariants_1_t_temp.append(Invariants_1_temp)
    Invariants_1.append(Invariants_1_t_temp)


# Experiment 2

Invariants_2 = []
for t in range(len(F_2)):
    Invariants_2_t_temp = []
    for x in range(len(F_2[0])):
        Invariants_2_temp = []
        for y in range(len(F_2[0][0])):
            F_2_array_temp = numpy.asarray(F_2[t][x][y])
            C = numpy.dot(numpy.transpose(F_2_array_temp), F_2_array_temp)
            I_1 = numpy.matrix.trace(C)
            I_4 = numpy.matrix.trace(numpy.dot(C, N1))
            I_6 = numpy.matrix.trace(numpy.dot(C, N2))
            Invariants_2_temp.append([I_1, I_4, I_6])
        Invariants_2_t_temp.append(Invariants_2_temp)
    Invariants_2.append(Invariants_2_t_temp)
    
    
# Experiment 3

Invariants_3 = []
for t in range(len(F_3)):
    Invariants_3_t_temp = []
    for x in range(len(F_3[0])):
        Invariants_3_temp = []
        for y in range(len(F_3[0][0])):
            F_3_array_temp = numpy.asarray(F_3[t][x][y])
            C = numpy.dot(numpy.transpose(F_3_array_temp), F_3_array_temp)
            I_1 = numpy.matrix.trace(C)
            I_4 = numpy.matrix.trace(numpy.dot(C, N1))
            I_6 = numpy.matrix.trace(numpy.dot(C, N2))
            Invariants_3_temp.append([I_1, I_4, I_6])
        Invariants_3_t_temp.append(Invariants_3_temp)
    Invariants_3.append(Invariants_3_t_temp)


# =============================================================================================================
# Neural Network part
# =============================================================================================================

# Determination of training data for every time step (t=0, ..., t=11)

Input_data = []
for n in range(3):
    if n==0:
        liste_Invariants = Invariants_1
    if n==1:
        liste_Invariants = Invariants_2
    if n==2:
        liste_Invariants = Invariants_3
    for t in range(len(liste_Invariants)):
        c = 0
        Input_data_1_temp = 0
        Input_data_2_temp = 0
        Input_data_3_temp = 0
        for x in range(len(liste_Invariants[0])):
            for y in range(len(liste_Invariants[0][0])):
                c+=1
                Input_data_1_temp+=liste_Invariants[t][x][y][0]
                Input_data_2_temp+=liste_Invariants[t][x][y][1]
                Input_data_3_temp+=liste_Invariants[t][x][y][2]
        Input_data.append([Input_data_1_temp/c, Input_data_2_temp/c, Input_data_3_temp/c])
    
    
Output_data = []
for n in range(3):
    if n==0:
        liste_Psi = Psi_List_1
    if n==1:
        liste_Psi = Psi_List_2
    if n==2:
        liste_Psi = Psi_List_3  
    for t in range(len(liste_Psi)):
        Output_data.append(liste_Psi[t])


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
        

# Model NN

model = keras.Sequential()

nb_neuron_1 = 16
nb_neuron_2 = 4

Activation = keras.activations.sigmoid

Epochs = 4000
Batchsize = 2

model.add(keras.Input(shape=(3*1)))
model.add(layers.Dense(nb_neuron_1, activation = Activation))
#model.add(layers.Dense(nb_neuron_2, activation = Activation))
model.add(layers.Dense(1))

inputs = keras.Input(shape=(3*1))
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

x_plot = []
for i in range(len(x_validation_sort)):
    x_plot.append(i)
    

plt.plot(x_plot, y_validation_plot, 'r')
plt.plot(x_plot, predictions_plot, 'b')
#plt.title('NN stress predictions for '+str(Epochs)+' epochs with '+str(nb_neuron_1)+' neurons on the first layer and '+str(nb_neuron_2)+' on the second layer', fontsize=20)
plt.title('NN psi predictions for '+str(Epochs)+' epochs with '+str(nb_neuron_1)+' neurons on the first layer', fontsize=20)
plt.xlabel('Triplets of invariants', fontsize=15)
plt.ylabel('Psi', fontsize=15)
plt.legend(['Abaqus results', 'NN predictions'], loc='lower right', fontsize=18)
plt.show()

# plt.plot(history.history['loss'])
# plt.show()
# print(min(history.history['loss']))

print("--- %s seconds ---" % (time.time() - start_time))