#Runs Monte Carlo sampling to simulate Annealing
# Most used arguments are -C, -S, and --Seed.
#Use --constant to to not change Beta, but keep it at BEnd
#Output files:
#Success file: success rate at particular annealing rate
#Run file: list of confiuration path that monte carlo took when sampling
#Stats file: Aceptance Rate, Minimum Found, Ground State

import tfim
import math
import numpy as np
import argparse
import Annealing as ann
import os
import cProfile
np.set_printoptions(precision=15)
np.set_printoptions(threshold=np.inf)






def main():

    #Add Arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('-N', type=int,default = 16, help = 'Number of spins')

    parser.add_argument('-C', type = int, default =1, help = 'J_Matrix confgiuration number')

    parser.add_argument('-S', type = int, default = 500, help = 'Annealing Rate in number of samples')

    parser.add_argument('--Seed', type = int, default = 3, help = "Random number seed")

    parser.add_argument('--EQ',type = int,default = 0, help = 'Equilibration time for Monte Carlo Sampling')

    parser.add_argument('--BStart', type = int,default = 0.1, help = "Starting temperature Beta")

    parser.add_argument('--BEnd', type = int, default = 10, help = "Ending temperature Beta")

    parser.add_argument('--Constant', type = bool, default = False, help = "Constant Temperature Sampling? (True/False)")

    args = parser.parse_args()



    #Parse arguments
    num_of_spins = args.N
    configuration = args.C
    Sample_Size = args.S
    seed = args.Seed
    J_BetaS = args.BStart
    J_BetaE = args.BEnd
    EQTime = args.EQ
    Constant = args.Constant




    #Load J Matrix configuration/ initiate basis
    J_Matrix = np.loadtxt("J_Matrices/J"+str(num_of_spins)+"/J_Matrix"+str(configuration)+"/"+str(num_of_spins)+"J_Matrix"+str(configuration)+".dat")
    lattice = tfim.Lattice([num_of_spins],True)
    basis = tfim.IsingBasis(lattice)


    #Deal with where to put files
    AnnDirectory = "J_Matrices/J"+str(num_of_spins)+"/J_Matrix"+str(configuration)+"/AnnRate"+str(Sample_Size)
    SeedDirectory = AnnDirectory + "/Seed" + str(seed)

    if not os.path.isdir(AnnDirectory):
        os.mkdir(AnnDirectory)
        os.mkdir(SeedDirectory)
    elif not os.path.isdir(SeedDirectory):
        os.mkdir(SeedDirectory)

    if Constant:
        RunFile = SeedDirectory + "/ConstMCSamples.dat"
        StatsFile = SeedDirectory + "/ConstMCStats.dat"
        SuccessFile = AnnDirectory + "/ConstSuccessRate.dat"
    else:
        RunFile = SeedDirectory + "/AnnMCSamples.dat"
        StatsFile = SeedDirectory + "/AnnMCStats.dat"
        SuccessFile = AnnDirectory + "/AnnSuccessRate.dat"



    minimum = np.loadtxt("J_Matrices/J"+str(num_of_spins)+"/J_Matrix"+str(configuration)+"/Minima.dat")[-1]

    success_numbers = np.array([[seed,0]])



    #Successfile is a running log, it's updated every time a different seed is run
    if os.path.exists(SuccessFile):
        seed_array = np.loadtxt(SuccessFile,ndmin = 2)


        duplicate = False
        i=0
        while i < seed_array.shape[0]:
            if seed_array[i][0] == seed:
                duplicate = True
                break
            i+=1
        if not duplicate:
            foundmin = ann.Monte_Carlo(lattice,basis,J_BetaS,J_BetaE,Sample_Size,seed,EQTime,J_Matrix,RunFile,StatsFile,Constant)
            if foundmin==minimum:
                success_numbers[0][1] = 1
            seed_array = np.concatenate((seed_array,success_numbers))

    else:
        foundmin = ann.Monte_Carlo(lattice,basis,J_BetaS,J_BetaE,Sample_Size,seed,EQTime,J_Matrix,RunFile,StatsFile,Constant)
        if foundmin==minimum:
            success_numbers[0][1] = 1
        seed_array = success_numbers



    header = str(np.sum(seed_array,axis=0)[1] / seed_array.shape[0])

    np.savetxt(SuccessFile,seed_array,delimiter = "\t", header = header)


if __name__ == "__main__":
    main()
