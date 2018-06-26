import tfim
import math
import numpy as np
import bisect
import progressbar
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
    fileU = open("EnergiesPerUpdate.txt","w")
    fileS = open("EnergyVsSampleSize.txt","w")
    fileD = open("StandardDeviationEnergy.txt","w")
    cumulative = np.cumsum(_Probability_Array)
    NN_Energy_Array = np.zeros(Sample_Size)
    for i in range(1,Sample_Size):
        r = np.random.rand()
        index = bisect.bisect_right(cumulative,r)
        Energy_Val = _Energy_Array[index]
        np.put(NN_Energy_Array,i,Energy_Val)
        if i%10 == 0:
            fileU.write(str(sum(NN_Energy_Array[i-10:i+1]) / 10)+"\n")
        fileS.write(str(sum(NN_Energy_Array[:i+1]) / i) + "\n")
        fileD.write(str(np.std(NN_Energy_Array[:i+1] / np.sqrt(i))) + "\n")
    fileU.close()
    fileS.close()
    fileD.close()
    return sum(NN_Energy_Array) / Sample_Size

def Tower_Sample_Average_Magnetization_Sqrd(_Probability_Array,Magnetization_Sqrd_Array,Sample_Size):
    fileU = open("MagPerUpdate.txt","w")
    fileS = open("MagVsSampleSize.txt","w")
    fileD = open("StandardDeviationMag.txt","w")
    cumulative = np.cumsum(_Probability_Array)
    NN_Mag_Array = np.zeros(Sample_Size)
    for i in range(1,Sample_Size):
        r = np.random.rand()
        index = bisect.bisect_right(cumulative,r)
        Mag_Val = Magnetization_Sqrd_Array[index]
        np.put(NN_Mag_Array,i,Mag_Val)
        if i%10 == 0:
            fileU.write(str(sum(NN_Mag_Array[i-10:i+1]) / 10)+"\n")
        fileS.write(str(sum(NN_Mag_Array[:i+1]) / i) + "\n")
        fileD.write(str(np.std(NN_Mag_Array[:i+1] / np.sqrt(i))) + "\n")
    fileU.close()
    fileS.close()
    fileD.close()
    return sum(NN_Mag_Array) / Sample_Size


lattice = tfim.Lattice([15],True)
basis = tfim.IsingBasis(lattice)
_Energy_Array = Energy_Array(lattice,basis)
_Probability_Array = Probability_Array(basis,_Energy_Array,1)
_Magnetization_Sqrd_Array = Magnetization_Sqrd_Array(basis)


fileAv = open("ExactAverage.txt","w")
fileAv.write(str(Average_Energy(basis,_Energy_Array,_Probability_Array)))
fileAv.close()

fileAv = open("ExactAverageMag.txt","w")
fileAv.write(str(Average_Magnetization_Sqrd(basis,_Probability_Array,_Magnetization_Sqrd_Array)))
fileAv.close()



Tower_Sample_Average_Magnetization_Sqrd(_Probability_Array,_Magnetization_Sqrd_Array,10000)
Tower_Sample_Average_Energy(_Probability_Array,_Energy_Array,10000)
