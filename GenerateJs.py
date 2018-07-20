import tfim
import numpy as np

num_of_spins = 25


J = tfim.Jij_instance(num_of_spins,1,"bimodal")

print(J)


np.savetxt("J_Matrix.dat",J)

find_min = False


fileU = open("JMins.txt")
if find_min:
    Min_Energy = infdim_State_Energy(lattice,basis,0,J_Matrix)/num_of_spins
    Min_State_Array = np.array([0])
    bar = progressbar.ProgressBar()
    for i in bar(range(2**25)):
        Energy = infdim_State_Energy(lattice,basis,i,J_Matrix)/num_of_spins
        if Energy<Min_Energy:
            Min_Energy = Energy
            Min_State_Array = np.array([i])
        elif Energy==Min_Energy:
            Min_State_Array = np.append(Min_State_Array,i)

    np.savetxt(fileU,Min_Energy)
    np.savetxt(fileU,Min_State_Array)
    np.savetxt(fileU,"Degeneracy: " + str(Min_State_Array.size))


# -3.44
#
# [ 6931573 13136997 15287397 15320165 15320181 15320437 18233994 18234250
#  18234266 18267034 20417434 26622858]
#
#  Degeneracy: 12
