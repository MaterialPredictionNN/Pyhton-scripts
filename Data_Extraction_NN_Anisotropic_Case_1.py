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

o1 = session.openOdb(name='C:/WINDOWS/system32/Job_Anisotropic_1.odb', readOnly=False)
odb = session.odbs['C:/WINDOWS/system32/Job_Anisotropic_1.odb']


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
    outputVariableName='External work: ALLWK for Whole Model', steps=('Step-1',),
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

with open('U1_List_1.txt', 'w') as file:
    for item in U1_List:
        file.write("%s,\n" % item)

U2_List = [] 
for i in range(len(U2[0])):
    U2_List_temp = []
    for j in range(len(U2)):
        U2_List_temp.append(U2[j][i][1])
    U2_List.append(U2_List_temp)

with open('U2_List_1.txt', 'w') as file:
    for item in U2_List:
        file.write("%s,\n" % item)

Stress_List = [] 
for i in range(len(S[0])):
    S_List_temp = []
    for j in range(len(S)):
        S_List_temp.append(S[j][i][1])
    Stress_List.append(S_List_temp)

with open('Stress_List_1.txt', 'w') as file:
    for item in Stress_List:
        file.write("%s,\n" % item)

External_Energy_List = []
for i in range(len(E)):
    External_Energy_List.append(E[i][1])

with open('External_Energy_List_1.txt', 'w') as file:
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

with open('X_List_1.txt', 'w') as file:
    for item in X_List:
        file.write("%s,\n" % item)

Y_List = [] 
for i in range(len(CoordY[0])):
    Y_List_temp = []
    for j in range(len(CoordY)):
        Y_List_temp.append(CoordY[j][i][1])
    Y_List.append(Y_List_temp)

with open('Y_List_1.txt', 'w') as file:
    for item in Y_List:
        file.write("%s,\n" % item)
