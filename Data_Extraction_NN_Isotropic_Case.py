# =============================================================================================================
# Imports 
# =============================================================================================================

from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
from odbAccess import *
import numpy

o1 = session.openOdb(name='C:/WINDOWS/system32/Uniaxial-Iso-Job1.odb', readOnly=False)
odb = session.odbs['C:/WINDOWS/system32/Uniaxial-Iso-Job1.odb']


# =============================================================================================================
# Extraction of stress and displacement
# =============================================================================================================

U1 = session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('U', 
    NODAL, ((COMPONENT, 'U1'), )), ), nodeSets=("DATA", ))

U2 = session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('U', 
    NODAL, ((COMPONENT, 'U2'), )), ), nodeSets=("DATA", ))

S = session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('S', 
    INTEGRATION_POINT, ((COMPONENT, 'S11'), )), ), nodeSets=("DATA", ))

E = session.XYDataFromHistory(name='ALLWK Whole Model-1', odb=odb,
    outputVariableName='External work: ALLWK for Whole Model', steps=('Uniaxial Tensile test',),
    __linkedVpName__='Viewport: 1')

# =============================================================================================================
# Export results into txt files for stress and displacement
# =============================================================================================================

U1_List = [] 
for i in range(len(U1[0])):
    U1_List_temp = []
    for j in range(len(U1)):
        U1_List_temp.append(U1[j][i][1])
    U1_List.append(U1_List_temp)

with open('U1_List.txt', 'w') as file:
    for item in U1_List:
        file.write("%s,\n" % item)

U2_List = [] 
for i in range(len(U2[0])):
    U2_List_temp = []
    for j in range(len(U2)):
        U2_List_temp.append(U2[j][i][1])
    U2_List.append(U2_List_temp)

with open('U2_List.txt', 'w') as file:
    for item in U2_List:
        file.write("%s,\n" % item)

Stress_List = [] 
for i in range(len(S[0])):
    S_List_temp = []
    for j in range(len(S)):
        S_List_temp.append(S[j][i][1])
    Stress_List.append(S_List_temp)

with open('Stress_List.txt', 'w') as file:
    for item in Stress_List:
        file.write("%s,\n" % item)

External_Energy_List = []
for i in range(len(E)):
    External_Energy_List.append(E[i][1])

with open('External_Energy_List.txt', 'w') as file:
    for item in External_Energy_List:
        file.write("%s,\n" % item)


# =============================================================================================================
# Extraction of nodal coordinates
# =============================================================================================================

CoordX = session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('COORD',NODAL, ((COMPONENT, 'COOR1'),  )), ), nodeSets=("DATA", ))

CoordY = session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('COORD',NODAL, ((COMPONENT, 'COOR2'),  )), ), nodeSets=("DATA", ))


# =============================================================================================================
# Export results into txt files for coordinates
# =============================================================================================================

X_List = [] 
for i in range(len(CoordX[0])):
    X_List_temp = []
    for j in range(len(CoordX)):
        X_List_temp.append(CoordX[j][i][1])
    X_List.append(X_List_temp)

with open('X_List.txt', 'w') as file:
    for item in X_List:
        file.write("%s,\n" % item)

Y_List = [] 
for i in range(len(CoordY[0])):
    Y_List_temp = []
    for j in range(len(CoordY)):
        Y_List_temp.append(CoordY[j][i][1])
    Y_List.append(Y_List_temp)

with open('Y_List.txt', 'w') as file:
    for item in Y_List:
        file.write("%s,\n" % item)


# =============================================================================================================
# Computation of matrix representation nodes indices
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


Nodes_Indices = []
indice = 0
for i in range(len(x0_sort)):
    Nodes_Indices_temp = []
    for j in range(len(y0_sort)):
        for k in range(len(x0)):
            if x0[k]==x0_sort[i] and y0[k]==y0_sort[j]:
                indice = k
        Nodes_Indices_temp.append(indice)
    Nodes_Indices.append(Nodes_Indices_temp)


# =============================================================================================================
# Gradient deformation matrix computation
# =============================================================================================================

F = []              # 3D Matrix as Matrix[t dimension][x dimensions][y dimension]
for t in range(len(X_List)):
    F_t_temp = []
    for x in range(1, len(Nodes_Indices)-1):
        F_temp = []
        for y in range(1, len(Nodes_Indices[0])-1):
            # if x==0:
            #     u1_1 = U1_List[t][Nodes_Indices[x+1][y]]
            #     u1_2 = U1_List[t][Nodes_Indices[x][y]]
            #     xh = 2*(X_List[t][Nodes_Indices[x+1][y]] - X_List[t][Nodes_Indices[x][y]])
            # if x==len(Nodes_Indices)-1:
            #     u1_1 = U1_List[t][Nodes_Indices[x][y]]
            #     u1_2 = U1_List[t][Nodes_Indices[x-1][y]]
            #     xh = 2*(X_List[t][Nodes_Indices[x][y]] - X_List[t][Nodes_Indices[x-1][y]])
            # if x!=0 and x!=len(Nodes_Indices)-1:
            #     u1_1 = U1_List[t][Nodes_Indices[x+1][y]]
            #     u1_2 = U1_List[t][Nodes_Indices[x-1][y]]
            #     xh = 2*(X_List[t][Nodes_Indices[x+1][y]] - X_List[t][Nodes_Indices[x-1][y]])
            # if y==0:
            #     u2_1 = U1_List[t][Nodes_Indices[x][y+1]]
            #     u2_2 = U2_List[t][Nodes_Indices[x][y]]
            #     yh = 2*(Y_List[t][Nodes_Indices[x][y+1]] - Y_List[t][Nodes_Indices[x][y]])
            # if y==len(Nodes_Indices[0])-1:
            #     u2_1 = U1_List[t][Nodes_Indices[x][y]]
            #     u2_2 = U2_List[t][Nodes_Indices[x][y-1]]
            #     yh = 2*(Y_List[t][Nodes_Indices[x][y]] - Y_List[t][Nodes_Indices[x][y-1]])
            # if y!=0 and y!=len(Nodes_Indices[0])-1:
            #     u2_1 = U1_List[t][Nodes_Indices[x][y+1]]
            #     u2_2 = U2_List[t][Nodes_Indices[x][y-1]]
            #     yh = 2*(Y_List[t][Nodes_Indices[x][y+1]] - Y_List[t][Nodes_Indices[x][y-1]])


            u1_1 = U1_List[t][Nodes_Indices[x+1][y]]
            u1_2 = U1_List[t][Nodes_Indices[x-1][y]]
            xh = 2*(X_List[t][Nodes_Indices[x+1][y]] - X_List[t][Nodes_Indices[x-1][y]])

            u2_1 = U1_List[t][Nodes_Indices[x][y+1]]
            u2_2 = U2_List[t][Nodes_Indices[x][y-1]]
            yh = 2*(Y_List[t][Nodes_Indices[x][y+1]] - Y_List[t][Nodes_Indices[x][y-1]])

            du1 = u1_1 - u1_2
            du2 = u2_1 - u2_2
            F11 = 1+du1/xh
            F12 = du1/yh
            F21 = du2/xh
            F22 = 1+du2/yh
            Fzz = 1/(F11*F22-F12*F21)
            F_temp.append([[F11, F12, 0],[F21, F22, 0], [0, 0, Fzz]])
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
            II_C = numpy.matrix.trace(numpy.dot(C,C))
            III_C = numpy.linalg.det(C)
            Invariants_temp.append([I_C,II_C,III_C])
        Invariants_t_temp.append(Invariants_temp)
    Invariants.append(Invariants_t_temp)


# =============================================================================================================
# Data for training creation
# =============================================================================================================

# for isotropic only
Strain_data = []
for t in range(len(Invariants)):
    Strain_data_t_temp = []
    for x in range(len(Invariants[0])):
        Strain_data_temp = []
        for y in range(len(Invariants[0][0])):
            Strain_data_temp.append(Invariants[t][x][y])
        Strain_data_t_temp.append(Strain_data_temp)
    Strain_data.append(Strain_data_t_temp)

Stress_data = []
for t in range(len(Stress_List)):
    Stress_data_t_temp = []
    for x in range(1, len(Nodes_Indices)-1):
        Stress_data_temp = []
        for y in range(1, len(Nodes_Indices[0])-1):
            Stress_data_temp.append(Stress_List[t][Nodes_Indices[x][y]])
        Stress_data_t_temp.append(Stress_data_temp)
    Stress_data.append(Stress_data_t_temp)


# =============================================================================================================
# Plot for verification
# =============================================================================================================

### Plot u, v
# u=[]
# v=[]
# u_x=[]
# u_y=[]
# v_x=[]
# v_y=[]
# t=10
# x_plot=[]
# y_plot=[]



# for x in range(len(F[t])):
#     for y in range(len(F[t][x])):
#         u.append(U1_List[t][Nodes_Indices[x][y]])
#         v.append(U2_List[t][Nodes_Indices[x][y]])
        
#         u_x.append(F[t][x][y][0][0]-1)
#         u_y.append(F[t][x][y][0][1])
#         v_x.append(F[t][x][y][1][0])
#         v_y.append(F[t][x][y][1][1]-1)
        
#         x_plot.append(X_List[t][Nodes_Indices[x][y]])
#         y_plot.append(Y_List[t][Nodes_Indices[x][y]])


# plt.scatter(y_plot, v_y)
# plt.title('v_y derivative plotted in function of Y coordinates at end time')
# plt.xlabel('Y coordinates')
# plt.ylabel('v_y derivative')
# #plt.axis([0,8,0.166,0.167])
# plt.show()