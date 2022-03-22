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
Y_List_1 = extract_list_from_text(r'C:\Windows\System32\Y_List_1.txt')
Y_List_2 = extract_list_from_text(r'C:\Windows\System32\Y_List_2.txt')
Y_List_3 = extract_list_from_text(r'C:\Windows\System32\Y_List_3.txt')
      

# x displacement data
U1_List_1 = extract_list_from_text(r'C:\Windows\System32\U1_List_1.txt')
U1_List_2 = extract_list_from_text(r'C:\Windows\System32\U1_List_2.txt')
U1_List_3 = extract_list_from_text(r'C:\Windows\System32\U1_List_3.txt')


# y displacement data
U2_List_1 = extract_list_from_text(r'C:\Windows\System32\U2_List_1.txt')
U2_List_2 = extract_list_from_text(r'C:\Windows\System32\U2_List_2.txt')
U2_List_3 = extract_list_from_text(r'C:\Windows\System32\U2_List_3.txt')


# external energy data
External_Energy_List_1 = extract_list_from_text(r'C:\Windows\System32\External_Energy_List_1.txt')
External_Energy_List_2 = extract_list_from_text(r'C:\Windows\System32\External_Energy_List_2.txt')
External_Energy_List_3 = extract_list_from_text(r'C:\Windows\System32\External_Energy_List_3.txt')


# Dimension

x_dim = 20.0
y_dim = 20.0
z_dim = 2.0


# Fibers orientation

gamma = 49.98
A = numpy.asarray([math.cos(gamma), math.sin(gamma), 0.0])
B = numpy.asarray([math.cos(gamma), -math.sin(gamma), 0.0])

global N1, N2
N1 = [[A[0]**2, A[0]*A[1], 0.0], [A[0]*A[1], A[1]**2, 0.0], [0.0, 0.0, 0.0]]
N2 = [[B[0]**2, B[0]*B[1], 0.0], [B[0]*B[1], B[1]**2, 0.0], [0.0, 0.0, 0.0]]

# psi energy
v_tot = x_dim*y_dim*z_dim
Psi_List_1 = []
Psi_List_2 = []
Psi_List_3 = []
for i in range(len(External_Energy_List_1)):
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


def deformation_gradient(U_Matrix, V_Matrix, X_Matrix, Y_Matrix):
    result = []
    for t in range(len(U_Matrix)):
        result_t_temp = []
        for x in range(1, len(U_Matrix[0])-1):
            result_temp = []
            for y in range(1, len(U_Matrix[0][0])-1):
                
                du_dx = (U_Matrix[t][x+1][y]-U_Matrix[t][x-1][y])/(X_Matrix[0][x+1][y]-X_Matrix[0][x-1][y])
                
                dv_dx = (V_Matrix[t][x+1][y]-V_Matrix[t][x-1][y])/(X_Matrix[0][x+1][y]-X_Matrix[0][x-1][y])
                
                du_dy = (U_Matrix[t][x][y+1]-U_Matrix[t][x][y-1])/(Y_Matrix[0][x][y+1]-Y_Matrix[0][x][y-1])
                
                dv_dy = (V_Matrix[t][x][y+1]-V_Matrix[t][x][y-1])/(Y_Matrix[0][x][y+1]-Y_Matrix[0][x][y-1])
                
                F11 = 1+du_dx
                F12 = du_dy
                F21 = dv_dx
                F22 = 1+dv_dy
                Fzz = 1/(F11*F22-F12*F21)
                result_temp.append([[F11, F12, 0], [F21, F22, 0], [0, 0, Fzz]])
            result_t_temp.append(result_temp)
        result.append(result_t_temp)    
    return result
    
# Experiment 1  

U_Matrix_1 = list2matrix(U1_List_1)
V_Matrix_1 = list2matrix(U2_List_1)
X_Matrix_1 = list2matrix(X_List_1)
Y_Matrix_1 = list2matrix(Y_List_1)

F_1 = deformation_gradient(U_Matrix_1, V_Matrix_1, X_Matrix_1, Y_Matrix_1)


# Experiment 2

U_Matrix_2 = list2matrix(U1_List_2)
V_Matrix_2 = list2matrix(U2_List_2)
X_Matrix_2 = list2matrix(X_List_2)
Y_Matrix_2 = list2matrix(Y_List_2)

F_2 = deformation_gradient(U_Matrix_2, V_Matrix_2, X_Matrix_2, Y_Matrix_2)


# Experiment 3  

U_Matrix_3 = list2matrix(U1_List_3)
V_Matrix_3 = list2matrix(U2_List_3)
X_Matrix_3 = list2matrix(X_List_3)
Y_Matrix_3 = list2matrix(Y_List_3)

F_3 = deformation_gradient(U_Matrix_3, V_Matrix_3, X_Matrix_3, Y_Matrix_3)


# =============================================================================================================
# Computation of the invariants (cf Bonnet & Wood - Nonlinear Continuum Mechanics for Finite Element Analysis)
# =============================================================================================================

def Invariants(F):
    result = []
    for t in range(len(F)):
        result_t_temp = []
        for x in range(len(F[0])):
            result_temp = []
            for y in range(len(F[0][0])):
                F_array_temp = numpy.asarray(F[t][x][y])
                C = numpy.dot(numpy.transpose(F_array_temp), F_array_temp)
                I_1 = numpy.matrix.trace(C)
                I_4 = numpy.matrix.trace(numpy.dot(C, N1))
                I_6 = numpy.matrix.trace(numpy.dot(C, N2))
                result_temp.append([I_1, I_4, I_6])
            result_t_temp.append(result_temp)
        result.append(result_t_temp)
    return result
    
# Experiment 1
Invariants_1 = Invariants(F_1)

# Experiment 2
Invariants_2 = Invariants(F_2)
    
# Experiment 3
Invariants_3 = Invariants(F_3)


# =============================================================================================================
# Neural Network part
# =============================================================================================================

# Determination of training data for every time step (t=0, ..., t=tf)

Input_data = []
for n in range(3):
    if n==0:
        list_Invariants = Invariants_1
    if n==1:
        list_Invariants = Invariants_2
    if n==2:
        list_Invariants = Invariants_3
    for t in range(len(list_Invariants)):
        c = 0
        Input_data_1_temp = 0
        Input_data_2_temp = 0
        Input_data_3_temp = 0
        for x in range(len(list_Invariants[0])):
            for y in range(len(list_Invariants[0][0])):
                c+=1
                Input_data_1_temp+=list_Invariants[t][x][y][0]
                Input_data_2_temp+=list_Invariants[t][x][y][1]
                Input_data_3_temp+=list_Invariants[t][x][y][2]
        Input_data.append([Input_data_1_temp/c, Input_data_2_temp/c, Input_data_3_temp/c])


Output_data = []
for n in range(3):
    if n==0:
        list_Psi = Psi_List_1
    if n==1:
        list_Psi = Psi_List_2
    if n==2:
        list_Psi = Psi_List_3  
    for t in range(len(list_Psi)):
        Output_data.append(list_Psi[t])

frac_train = 0.7
frac_test = 0.15
frac_validation = 0.15

n_test = 3
n_tot = len(Invariants_1)

l_indice_temp_test1 = [i for i in range(0, n_tot)]
l_indice_temp_test2 = [i for i in range(n_tot, 2*n_tot)]
l_indice_temp_test3 = [i for i in range(2*n_tot, 3*n_tot)]

random.shuffle(l_indice_temp_test1)
random.shuffle(l_indice_temp_test2)
random.shuffle(l_indice_temp_test3)

x_train = []
y_train = []

x_test = []
y_test = []

x_validation_test1 = []
x_validation_test2 = []
x_validation_test3 = []

y_validation_test1 = []
y_validation_test2 = []
y_validation_test3 = []

for i in range(n_tot):
    if i<frac_train*n_tot:
        x_train.append(Input_data[l_indice_temp_test1[i]])
        x_train.append(Input_data[l_indice_temp_test2[i]])
        x_train.append(Input_data[l_indice_temp_test3[i]])
        
        y_train.append(Output_data[l_indice_temp_test1[i]])
        y_train.append(Output_data[l_indice_temp_test2[i]])
        y_train.append(Output_data[l_indice_temp_test3[i]])
        
    if i>frac_train*n_tot and i<frac_train*n_tot+frac_test*n_tot:
        x_test.append(Input_data[l_indice_temp_test1[i]])
        x_test.append(Input_data[l_indice_temp_test2[i]])
        x_test.append(Input_data[l_indice_temp_test3[i]])
        
        y_test.append(Output_data[l_indice_temp_test1[i]])
        y_test.append(Output_data[l_indice_temp_test2[i]])
        y_test.append(Output_data[l_indice_temp_test3[i]])

    if i>frac_train*n_tot+frac_test*n_tot:
        x_validation_test1.append(Input_data[l_indice_temp_test1[i]])
        x_validation_test2.append(Input_data[l_indice_temp_test2[i]])
        x_validation_test3.append(Input_data[l_indice_temp_test3[i]])
        
        y_validation_test1.append(Output_data[l_indice_temp_test1[i]])
        y_validation_test2.append(Output_data[l_indice_temp_test2[i]])
        y_validation_test3.append(Output_data[l_indice_temp_test3[i]])
        

# Model NN

model = keras.Sequential()

nb_neuron_1 = 2
#nb_neuron_2 = 1

Activation = keras.activations.sigmoid

Epochs = 1000
Batchsize = 4

model.add(keras.Input(shape=(3*1)))
model.add(layers.Dense(nb_neuron_1, activation = Activation))
model.add(layers.Dense(1))

inputs = keras.Input(shape=(3*1))
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

def list_indice_sort(list_ini, list_sort):
    result = []
    for i in range(len(list_sort)):
        j=0
        while list_sort[i] != list_ini[j]:
            j+=1
        result.append(j)
    return result


# Test 1
predictions_test1 = model.predict(x_validation_test1)
x_validation_sort_test1 = sorted(x_validation_test1)

y_validation_plot_test1 = []
predictions_plot_test1 = []

list_temp = list_indice_sort(x_validation_test1, x_validation_sort_test1)
for i in range(len(list_temp)):
    indice = list_temp[i]
    y_validation_plot_test1.append(y_validation_test1[indice])
    predictions_plot_test1.append(predictions_test1[indice])

x_plot1_test1 = []
x_plot4_test1 = []
x_plot6_test1 = []
for i in range(len(x_validation_sort_test1)):
    x_plot1_test1.append(x_validation_sort_test1[i][0])
    x_plot4_test1.append(x_validation_sort_test1[i][1])
    x_plot6_test1.append(x_validation_sort_test1[i][2])
    

# Test 2
predictions_test2 = model.predict(x_validation_test2)
x_validation_sort_test2 = sorted(x_validation_test2)

y_validation_plot_test2 = []
predictions_plot_test2 = []

list_temp = list_indice_sort(x_validation_test2, x_validation_sort_test2)
for i in range(len(list_temp)):
    indice = list_temp[i]
    y_validation_plot_test2.append(y_validation_test2[indice])
    predictions_plot_test2.append(predictions_test2[indice])

x_plot1_test2 = []
x_plot4_test2 = []
x_plot6_test2 = []
for i in range(len(x_validation_sort_test2)):
    x_plot1_test2.append(x_validation_sort_test2[i][0])
    x_plot4_test2.append(x_validation_sort_test2[i][1])
    x_plot6_test2.append(x_validation_sort_test2[i][2])


# Test 3
predictions_test3 = model.predict(x_validation_test3)
x_validation_sort_test3 = sorted(x_validation_test3)

y_validation_plot_test3 = []
predictions_plot_test3 = []

list_temp = list_indice_sort(x_validation_test3, x_validation_sort_test3)
for i in range(len(list_temp)):
    indice = list_temp[i]
    y_validation_plot_test3.append(y_validation_test3[indice])
    predictions_plot_test3.append(predictions_test3[indice])

x_plot1_test3 = []
x_plot4_test3 = []
x_plot6_test3 = []
for i in range(len(x_validation_sort_test3)):
    x_plot1_test3.append(x_validation_sort_test3[i][0])
    x_plot4_test3.append(x_validation_sort_test3[i][1])
    x_plot6_test3.append(x_validation_sort_test3[i][2])
    

# Plot of psi as a function of the first invariant
plt.plot(x_plot1_test1, y_validation_plot_test1, 'r', linewidth=1)
plt.plot(x_plot1_test1, predictions_plot_test1, 'r:^', linewidth=1)

plt.plot(x_plot1_test2, y_validation_plot_test2, 'b', linewidth=1)
plt.plot(x_plot1_test2, predictions_plot_test2, 'b:^', linewidth=1)

plt.plot(x_plot1_test3, y_validation_plot_test3, 'm', linewidth=1)
plt.plot(x_plot1_test3, predictions_plot_test3, 'm:^', linewidth=1)

plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.title('ANN psi predictions for '+str(Epochs)+' epochs with '+str(nb_neuron_1)+' neurons on the first layer', fontsize=38)
plt.xlabel('First invariant', fontsize=40)
plt.ylabel('Psi', fontsize=40)
plt.legend(['Test 1 - data', 'Test 1 - prediction', 'Test 2 - data', 'Test 2 - prediction', 'Test 3 - data', 'Test 3 - prediction'], loc='upper left', fontsize=35)
plt.show()


# Plot of psi as a function of the fourth invariant
plt.plot(x_plot4_test1, y_validation_plot_test1, 'r', linewidth=1)
plt.plot(x_plot4_test1, predictions_plot_test1, 'r:^', linewidth=1)

plt.plot(x_plot4_test2, y_validation_plot_test2, 'b', linewidth=1)
plt.plot(x_plot4_test2, predictions_plot_test2, 'b:^', linewidth=1)

plt.plot(x_plot4_test3, y_validation_plot_test3, 'm', linewidth=1)
plt.plot(x_plot4_test3, predictions_plot_test3, 'm:^', linewidth=1)

plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.title('ANN psi predictions for '+str(Epochs)+' epochs with '+str(nb_neuron_1)+' neurons on the first layer', fontsize=38)
plt.xlabel('Fourth or sixth invariant', fontsize=35)
plt.ylabel('Psi', fontsize=35)
plt.legend(['Test 1 - data', 'Test 1 - prediction', 'Test 2 - data', 'Test 2 - prediction', 'Test 3 - data', 'Test 3 - prediction'], loc='upper left', fontsize=25)
plt.show()

# Plot of psi as a function of the first invariant
plt.plot(x_plot6_test1, y_validation_plot_test1, 'r', linewidth=1)
plt.plot(x_plot6_test1, predictions_plot_test1, 'r:^', linewidth=1)

plt.plot(x_plot6_test2, y_validation_plot_test2, 'b', linewidth=1)
plt.plot(x_plot6_test2, predictions_plot_test2, 'b:^', linewidth=1)

plt.plot(x_plot6_test3, y_validation_plot_test3, 'm', linewidth=1)
plt.plot(x_plot6_test3, predictions_plot_test3, 'm:^', linewidth=1)

plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.title('ANN psi predictions for '+str(Epochs)+' epochs with '+str(nb_neuron_1)+' neurons on the first layer', fontsize=38)
plt.xlabel('Fourth or sixth invariant', fontsize=40)
plt.ylabel('Psi', fontsize=40)
plt.legend(['Test 1 - data', 'Test 1 - prediction', 'Test 2 - data', 'Test 2 - prediction', 'Test 3 - data', 'Test 3 - prediction'], loc='upper left', fontsize=35)
plt.show()


# =============================================================================================================
# Error plot
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
# End
# =============================================================================================================

print("--- %s seconds ---" % (time.time() - start_time))


# =============================================================================================================
# Get weights and biases
# =============================================================================================================

weights_1 = model.layers[1].get_weights()[0]
biases_1 = model.layers[1].get_weights()[1]
weights_2 = model.layers[2].get_weights()[0]
biases_2 = model.layers[2].get_weights()[1]

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def relu(x):
    return(max(0, x))

w11 = weights_1[0][0]
w12 = weights_1[0][1]
#w13 = weights_1[0][2]
#w14 = weights_1[0][3]
w21 = weights_1[1][0]
w22 = weights_1[1][1]
#w23 = weights_1[1][2]
#w24 = weights_1[1][3]
w31 = weights_1[2][0]
w32 = weights_1[2][1]
#w33 = weights_1[2][2]
#w34 = weights_1[2][3]

b1 = biases_1[0]
b2 = biases_1[1]
#b3 = biases_1[2]
#b4 = biases_1[3]

wf1 = weights_2[0][0]
wf2 = weights_2[1][0]
#wf3 = weights_2[2][0]
#wf4 = weights_2[3][0]

bf = biases_2[0]

def nnresults(input_inv):
    I1 = input_inv[0]
    I4 = input_inv[1]
    I6 = input_inv[2]
    
    a1 = sigmoid(w11*I1 + w21*I4 + w31*I6 + b1)
    a2 = sigmoid(w12*I1 + w22*I4 + w32*I6 + b2)
    #a3 = relu(w13*I1 + w23*I4 + w33*I6 + b3)
    #a4 = relu(w14*I1 + w24*I4 + w34*I6 + b4)
    
    #psi = a1*wf1 + a2*wf2 + a3*wf3+ a4*wf4 + bf
    psi = a1*wf1 + a2*wf2 + bf
    
    return psi


# =============================================================================================================
# Derivative of psi
# =============================================================================================================

def g_func(k, input_inv):
    I1 = input_inv[0]
    I4 = input_inv[1]
    I6 = input_inv[2]
    if k == 0:
        return w11*I1 + w21*I4 + w31*I6 + b1
    if k == 1:
        return w12*I1 + w22*I4 + w32*I6 + b2
    else :
        return 0


def dpsi_dI(input_inv, indice_diff):
    result = 0
    for k in range(2):
        result+=weights_2[k][0]*weights_1[indice_diff][k]*sigmoid(-g_func(k, input_inv))*(1-sigmoid(-g_func(k, input_inv)))
    return result

# Validation on test 3

Input_test = []
for t in range(len(Invariants_3)):
    c = 0
    Input_data_1_temp = 0
    Input_data_2_temp = 0
    Input_data_3_temp = 0
    for x in range(len(Invariants_3[0])):
        for y in range(len(Invariants_3[0][0])):
            c+=1
            Input_data_1_temp+=Invariants_3[t][x][y][0]
            Input_data_2_temp+=Invariants_3[t][x][y][1]
            Input_data_3_temp+=Invariants_3[t][x][y][2]
    Input_test.append([Input_data_1_temp/c, Input_data_2_temp/c, Input_data_3_temp/c])

Input_test_1_I1 = []
Input_test_1_I4 = []
Input_test_1_I6 = []
for i in range(len(Input_test)):
    Input_test_1_I1.append(Input_test[i][0])
    Input_test_1_I4.append(Input_test[i][1])
    Input_test_1_I6.append(Input_test[i][2])
    
Output_test_1 = model.predict(Input_test)


# Derivative of psi in regard of I1
print("Derivative of psi in regard of I1")
indice_diff=0

formula_diff = []
for i in range(len(Input_test)):
    input_inv = Input_test[i]
    formula_diff.append(dpsi_dI(input_inv, indice_diff))
    
computed_diff = []
for i in range(1, len(Input_test)-1):
    dx = Input_test[i+1][indice_diff] - Input_test[i-1][indice_diff]
    dy = Output_test_1[i+1] - Output_test_1[i-1]
    computed_diff.append(dy/dx)
    

plt.plot(Input_test_1_I1[100:1000], formula_diff[100:1000], 'r')
#plt.plot(Input_test_1_I1[100:1000], computed_diff[99:999], 'b')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.title('Psi derivative in regard of I1', fontsize=38)
plt.xlabel('Firsth invariant', fontsize=35)
plt.ylabel('dpsi/dI1', fontsize=35)
plt.legend(['Analytic formula', 'Discrete calculation'], fontsize=25)
plt.show()


# Derivative of psi in regard of I4
print("Derivative of psi in regard of I4")
indice_diff=1

formula_diff = []
for i in range(len(Input_test)):
    input_inv = Input_test[i]
    formula_diff.append(dpsi_dI(input_inv, indice_diff))

computed_diff = []
for i in range(1, len(Input_test)-1):
    dx = Input_test[i+1][indice_diff] - Input_test[i-1][indice_diff]
    dy = Output_test_1[i+1] - Output_test_1[i-1]
    computed_diff.append(dy/dx)    

plt.plot(Input_test_1_I4[100:1000], formula_diff[100:1000], 'r')
#plt.plot(Input_test_1_I4[100:1000], computed_diff[99:999], 'b')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.title('Psi derivative in regard of I4', fontsize=38)
plt.xlabel('Fourth invariant', fontsize=35)
plt.ylabel('dpsi/dI4', fontsize=35)
plt.legend(['Analytic formula', 'Discrete calculation'], fontsize=25)
plt.show()


# Derivative of psi in regard of I6
print("Derivative of psi in regard of I6")
indice_diff = 2

formula_diff = []
for i in range(len(Input_test)):
    input_inv = Input_test[i]
    formula_diff.append(dpsi_dI(input_inv, indice_diff))
    
computed_diff = []
for i in range(1, len(Input_test)-1):
    dx = Input_test[i+1][indice_diff] - Input_test[i-1][indice_diff]
    dy = Output_test_1[i+1] - Output_test_1[i-1]
    computed_diff.append(dy/dx)
    

plt.plot(Input_test_1_I6[100:1000], formula_diff[100:1000], 'r')
#plt.plot(Input_test_1_I6[100:1000], computed_diff[99:999], 'b')
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.title('Psi derivative in regard of I6', fontsize=38)
plt.xlabel('Sixth invariant', fontsize=35)
plt.ylabel('dpsi/dI6', fontsize=35)
plt.legend(['Analytic formula', 'Discrete calculation'], fontsize=25)
plt.show()


# =============================================================================================================
# =============================================================================================================
# =============================================================================================================