import tfim
import math
import numpy as np
import bisect
np.set_printoptions(precision=15)


def State_Energy(lattice,basis,index):
    config = lattice.config(basis.state(index))
    _NN_config = lattice.NN_config(config,1)
    return sum(-1 * config * _NN_config)

def Energy_Array(lattice,basis):
    _Energy_Array = np.array([State_Energy(lattice,basis,i) for i in range(basis.M)])
    return _Energy_Array

def Probability_Array(basis,_Energy_Array,J_Beta):
    NN_Probability_Array = np.array([math.exp(-1 * J_Beta * _Energy_Array[i]) for i in range(basis.M)])
    Z = sum(NN_Probability_Array)
    _Probability_Array = np.array([NN_Probability_Array[i] / Z for i in range(basis.M)])
    return _Probability_Array

def Magnetization_Sqrd_Array(basis):
    return np.array([sum(basis.spin_state(i))**2 for i in range(basis.M)])


def Average_Energy(basis,_Energy_Array,_Probability_Array):
    return sum(_Energy_Array * _Probability_Array)

def Average_Magnetization_Sqrd(basis,_Probability_Array,_Magnetization_Sqrd_Array):
    return sum(_Probability_Array * _Magnetization_Sqrd_Array)

def J_Beta_Range_Energy(basis,_Energy_Array,start,end,step):
    return np.array([Average_Energy(basis,_Energy_Array,Probability_Array(basis,_Energy_Array,round(i,1))) for i in np.arange(start,end,step)])

def Tower_Sample_Average_Energy(_Probability_Array,_Energy_Array,Sample_Size):
    file = open("EnergiesPerUpdate.txt","w")
    cumulative = np.cumsum(_Probability_Array)
    NN_Energy_Array = np.zeros(Sample_Size)
    for i in range(Sample_Size):
        r = np.random.rand()
        index = bisect.bisect_right(cumulative,r)
        Energy_Val = _Energy_Array[index]
        file.write(str(Energy_Val)+"\n")
        np.put(NN_Energy_Array,i,Energy_Val)
    file.close()
    return sum(NN_Energy_Array) / Sample_Size

def Tower_Sample_Average_Magnetization_Sqrd(_Probability_Array,Magnetization_Sqrd_Array,Sample_Size):
    cumulative = np.cumsum(_Probability_Array)
    NN_Mag_Array = np.zeros(Sample_Size)
    for i in range(Sample_Size):
        r = np.random.rand()
        index = bisect.bisect_right(cumulative,r)
        Mag_Val = Magnetization_Sqrd_Array[index]
        np.put(NN_Mag_Array,i,Mag_Val)
    return sum(NN_Mag_Array) / Sample_Size


lattice = tfim.Lattice([5],True)
basis = tfim.IsingBasis(lattice)
_Energy_Array = Energy_Array(lattice,basis)
_Probability_Array = Probability_Array(basis,_Energy_Array,1)
#_Magnetization_Sqrd_Array = Magnetization_Sqrd_Array(basis)

#_Average_Magnetization_Sqrd = Average_Magnetization_Sqrd(basis,_Probability_Array,_Magnetization_Sqrd_Array)
_Average_Energy = Average_Energy(basis,_Energy_Array,_Probability_Array)
#_J_Beta_Range_Energy = J_Beta_Range_Energy(basis,_Energy_Array,0.1,10,0.2)

# print(_Energy_Array)
# print(_Probability_Array)
# print(_Magnetization_Sqrd_Array)
print(_Average_Energy)
#print(_Average_Magnetization_Sqrd)
# print(_J_Beta_Range_Energy)

print(Tower_Sample_Average_Energy(_Probability_Array,_Energy_Array,10000))
#print(Tower_Sample_Average_Magnetization_Sqrd(_Probability_Array,_Magnetization_Sqrd_Array,100000))
