#Takes Number of Spins and RNG seed (as ints)
#Saves configuration of spins in file [num_of_spins]J_Matrix[seed].dat
#this file can be read by MCAnnealing.py, Minima.py, and EnergyArray.py


import tfim
import numpy as np
from pathlib import Path
import os
import argparse


def main():
    #Parse Args
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int,default = 16, help = 'Number of spins')
    parser.add_argument('-Seed', type = int, default =1, help = 'RNG seed for generating J Matrices')
    parser.add_argument('-J', type = int, default = 1, help = 'Weight/Value of bonds between spins')
    args = parser.parse_args()

    #ParseArgs
    num_of_spins = args.N
    seed = args.Seed


    #Make a place to put the file
    root = Path(".")
    folder = os.getcwd() + "/J_Matrices/J" + str(num_of_spins)
    if not Path(folder).exists():
        os.mkdir(folder)

    #Generate random instance and save it
    J = tfim.Jij_instance(num_of_spins,1,"bimodal",seed)
    np.savetxt(folder + "/" + str(num_of_spins) + "J_Matrix" + str(seed) + ".dat",J)

if __name__ == "__main__":
    main()
