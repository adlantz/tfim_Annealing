import tfim
import math
import numpy as np
import bisect
import progressbar
np.set_printoptions(precision=15)
np.set_printoptions(threshold=np.inf)


##############################################################################
# EXACT ENERGY FUNCTIONS


def State_Energy(lattice,basis,index):
    """Returns the Energy of one instance (indicated by index) of a lattice"""
    config = lattice.config(basis.state(index))
    _NN_config = lattice.NN_config(config,1)
    return sum(-1 * config * _NN_config)

def infdim_State_Energy(lattice,basis,index,J_Matrix):
    Energy = 0
    shift_state = np.zeros(basis.N,dtype=int)
    state = basis.spin_state(index)

    for shift in range(1,basis.N//2+1):
        shift_state[shift:] = state[:-shift]
        shift_state[:shift] = state[-shift:]

        if (basis.N%2 == 0) and (shift == basis.N//2):
            Energy += 0.5*np.dot(J_Matrix[shift-1,:]*shift_state,state)

        else:
            Energy += np.dot(J_Matrix[shift-1,:]*shift_state,state)
    return Energy

def Energy_Array(lattice,basis):
    """Returns an array of the energy states for every possible configuration of
    a lattice"""
    _Energy_Array = np.array([State_Energy(lattice,basis,i) for i in range(basis.M)])
    return _Energy_Array

def NN_State_Probability(J_Beta,_State_Energy):
    """Returns the Non-normalized probability of an energy state ocurring
    NNP = e ^ (-E/kT)"""
    return math.exp(-1 * J_Beta * _State_Energy)

def NN_State_Log_Probability(J_Beta,_State_Energy):
    return -1 * J_Beta * _State_Energy

def Probability_Array(basis,_Energy_Array,J_Beta):
    """Returns an array of the probabilites of every possible configuration of
    a lattice"""
    NN_Probability_Array = np.array([NN_State_Probability(J_Beta,_Energy_Array[i]) for i in range(basis.M)])
    Z = sum(NN_Probability_Array)
    _Probability_Array = np.array([NN_Probability_Array[i] / Z for i in range(basis.M)])
    return _Probability_Array

def State_Magnetization_Sqrd(basis,index):
    """Returns the magnetization squared of a state indicated by index"""
    return sum(basis.spin_state(index))**2

def Magnetization_Sqrd_Array(basis):
    """Returns an array of the magnetizations squared of every possible
    configuration of a lattice"""
    return np.array([sum(basis.spin_state(i))**2 for i in range(basis.M)])


def Average_Energy(basis,_Energy_Array,_Probability_Array):
    """Returns the average Energy of lattice"""
    return sum(_Energy_Array * _Probability_Array) / num_of_spins


def Average_Magnetization_Sqrd(basis,_Probability_Array,_Magnetization_Sqrd_Array):
    """Returns average Magnetization Squared of lattice"""
    return sum(_Probability_Array * _Magnetization_Sqrd_Array) / num_of_spins


##############################################################################
# TOWER SAMPLING FUNCTIONS
def Tower_Sample_Energy(_Probability_Array,_Energy_Array,Sample_Size,seed):
    """Samples Energies based on their probability using tower sampling direct
    sampling method"""
    np.random.seed(seed)
    fileU = open("TSEnergiesPerUpdate.txt","w")
    fileU.write(str(Average_Energy(basis,_Energy_Array,_Probability_Array)) + "\t"
    + str(num_of_spins) + "\n")
    cumulative = np.cumsum(_Probability_Array)
    for i in range(Sample_Size):
        r = np.random.rand()
        index = bisect.bisect_right(cumulative,r)
        Energy_Val = _Energy_Array[index]
        fileU.write(str(Energy_Val/num_of_spins) + "\n")

        # fileU.write(str(sum(NN_Energy_Array[i-bin_size+1:i+1]) / bin_size) + "\n")
    fileU.close()

def Tower_Sample_Magnetization_Sqrd(_Probability_Array,Magnetization_Sqrd_Array,Sample_Size,seed):
    """Samples Magnetizations Squared based on their probability using tower
    sampling direct sampling method"""
    np.random.seed(seed)
    fileU = open("TSMagPerUpdate.txt","w")
    fileU.write(str(Average_Magnetization_Sqrd(basis,_Probability_Array,_Magnetization_Sqrd_Array)) + "\t"
    + str(num_of_spins) + "\n")
    cumulative = np.cumsum(_Probability_Array)
    for i in range(Sample_Size):
        r = np.random.rand()
        index = bisect.bisect_right(cumulative,r)
        Mag_Val = Magnetization_Sqrd_Array[index]
        fileU.write(str(Mag_Val/num_of_spins) + "\n")
        # fileU.write(str(sum(NN_Mag_Array[i-bin_size+1:i+1]) / bin_size) + "\n")
    fileU.close()


##############################################################################
# MARKOV CHAIN MONTE CARLO SAMPLING

def Monte_Carlo(lattice,basis,J_BetaS,J_BetaE,Sample_Size,seed,EQTime,J_Matrix,File,writefile=True,Seed=True,averages=True):
    """Generates Markov chain of Energies based on their relative
    probabilities"""
    fileU = open(File,"w")

    if averages:
        header = str(Average_Energy(basis,infdim_Energy_Array,_Probability_Array)) + "\t" + str(Average_Magnetization_Sqrd(basis,_Probability_Array,_Magnetization_Sqrd_Array)) + "\t" + str(num_of_spins)
    else:
        header = str(0) + "\t" + str(0) + "\t" + str(num_of_spins)

    J_BetaStep = (J_BetaE-J_BetaS)/Sample_Size

    fileU.write(header + "\n")
    Acceptance_Number = 0

    if(Seed):
        np.random.seed(seed)


    state = basis.state(np.random.randint(basis.M))
    J_Beta = J_BetaS

    bar = progressbar.ProgressBar()
    for i in bar(range(Sample_Size)):
        _State_Mag_Beta = State_Magnetization_Sqrd(basis,basis.index(state))
        _State_Energy_Beta = infdim_State_Energy(lattice,basis,basis.index(state),J_Matrix)
        Prob_Beta = NN_State_Log_Probability(J_Beta,_State_Energy_Beta)

        index = np.random.randint(0,state.size)
        basis.flip(state,index)

        _State_Energy_Alpha = infdim_State_Energy(lattice,basis,basis.index(state),J_Matrix)
        Prob_Alpha = NN_State_Log_Probability(J_Beta,_State_Energy_Alpha)

        probability_append = min(1,np.exp(Prob_Alpha - Prob_Beta))
        random = np.random.rand()

        if random <= probability_append:
            Acceptance_Number += 1
            _State_Mag_Alpha = State_Magnetization_Sqrd(basis,basis.index(state))
            if i >= EQTime:
                val_Array = np.array([[_State_Energy_Alpha,_State_Mag_Alpha]])/num_of_spins
                np.savetxt(fileU,val_Array,delimiter="\t")

        else:

            basis.flip(state,index)
            if i >= EQTime:
                val_Array = np.array([[_State_Energy_Beta,_State_Mag_Beta]])/num_of_spins
                np.savetxt(fileU,val_Array,delimiter="\t")

        J_Beta+=J_BetaStep
    print(Acceptance_Number*100/Sample_Size)
    fileU.close()


##############################################################################
# MAIN

J_Beta = 10
num_of_spins = 30
seed = np.random.randint(2**30)
EQTime = 250
J_Matrix = np.loadtxt("J_Matrix.dat")
lattice = tfim.Lattice([num_of_spins],True)
basis = tfim.IsingBasis(lattice)



print(J_Matrix)




# _Energy_Array = Energy_Array(lattice,basis)
# infdim_Energy_Array = tfim.JZZ_SK_ME(basis,J_Matrix)
# _Probability_Array = Probability_Array(basis,infdim_Energy_Array,J_Beta)
# _Magnetization_Sqrd_Array = Magnetization_Sqrd_Array(basis)

# print(Average_Magnetization_Sqrd(basis,_Probability_Array,_Magnetization_Sqrd_Array))
#
# Annealing = "MCPerUpdate.txt"
# Equilibration = "MCEqPerUpdate.txt"
#
#
#
# Monte_Carlo(lattice,basis,0.1,J_Beta,500000,seed,EQTime,J_Matrix,"MCPerUpdate1.txt",averages=False)
# # Monte_Carlo(lattice,basis,0.1,J_Beta,250000,seed,EQTime,J_Matrix,"MCPerUpdate2.txt",averages=False)
# # Monte_Carlo(lattice,basis,0.1,J_Beta,100000,seed,EQTime,J_Matrix,"MCPerUpdate3.txt",averages=False)
# Monte_Carlo(lattice,basis,J_Beta,J_Beta,500000,seed,EQTime,J_Matrix,Equilibration,averages=False)




# Monte_Carlo_Mag(lattice,basis,J_Beta,50000,seed,EQTime)
# Tower_Sample_Magnetization_Sqrd(_Probability_Array,_Magnetization_Sqrd_Array,25000,seed)
# Tower_Sample_Energy(_Probability_Array,_Energy_Array,10000,seed)
# Monte_Carlo_Energy(lattice,basis,J_Beta,5000,seed,EQTime)
