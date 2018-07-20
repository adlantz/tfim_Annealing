import tfim
import numpy as np
import Annealing as ann

num_of_spins = 30
lattice = tfim.Lattice([num_of_spins],True)
basis = tfim.IsingBasis(lattice)

#
# J = tfim.Jij_instance(num_of_spins,1,"bimodal")

J_Matrix = np.loadtxt("J_Matrix.dat")

find_min = True



if find_min:
    fileU = open("JMins.txt","w")
    Min_Energy = ann.infdim_State_Energy(lattice,basis,0,J_Matrix)/num_of_spins
    Min_State_Array = np.array([0])
    bar = progressbar.ProgressBar()
    for i in bar(range(2**25)):
        Energy = ann.infdim_State_Energy(lattice,basis,i,J_Matrix)/num_of_spins
        if Energy<Min_Energy:
            Min_Energy = Energy
            Min_State_Array = np.array([i])
        elif Energy==Min_Energy:
            Min_State_Array = np.append(Min_State_Array,i)

    fileU.write(Min_Energy)
    np.savetxt(fileU,Min_State_Array,delimiter="/t")
    fileU.write("\nDegeneracy: " + str(Min_State_Array.size))
    print(Min_Energy)
    print(Min_State_Array)
    print("Degeneracy: " + str(Min_State_Array.size))
    fileU.close()


# -3.44
#
# [ 6931573 13136997 15287397 15320165 15320181 15320437 18233994 18234250
#  18234266 18267034 20417434 26622858]
#
#  Degeneracy: 12
