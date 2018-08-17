import tfim
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int,default = 16, help = 'Number of spins')
    parser.add_argument('-C', type = int, default = 3, help = 'J Matrix configuration')
    args = parser.parse_args()

    num_of_spins = args.N
    configuration = args.C
    J_Matrix = np.loadtxt("J_Matrices/J"+str(num_of_spins)+"/J_Matrix"+str(configuration)+"/"+str(num_of_spins)+"J_Matrix"+str(configuration)+".dat")
    lattice = tfim.Lattice([num_of_spins],True)
    basis = tfim.IsingBasis(lattice)
    infdim_Energy_Array = tfim.JZZ_SK_ME(basis,J_Matrix)

    np.savetxt("J_Matrices/J"+str(num_of_spins)+"/J_Matrix"+str(configuration)+"/EnergyArray.dat",infdim_Energy_Array)



if __name__ == "__main__":
    main()
