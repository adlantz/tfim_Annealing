# Classical Ising Model Annealing

 Simulating classical annealing on an infinite dimensional lattice of spin states us Markov Chain Monte Carlo methods.

## Overview

GenerateJs.py creates random instances of an infinite dimensional spin configuration. EnergyArray.py and Minima.py. then allow you to manually find all of the energy states and the minima of the energy states. Using this MCAnnealing.py uses functions from Annealing.py to use Monte Carlo sampling methods to simulate annealing on a specific configuration, returning the path of states that the sampling method took, and the minima it found.

## Standard Procedure

### Generating J Matrices

The file GenerateJs.py takes arguments
```
  -h, --help  show this help message and exit
  -N N        Number of spins
  -Seed SEED  RNG seed for generating J Matrices
  -J J        Weight/Value of bonds between spins
```
 This generates one J Matrix in a .dat file.This .dat file is then read in by other files in this repository.

### Finding Energy Array and Minima

EnergyArray.py and minima.py create .dat files of an array of all possible energy values and the minima in that array, respectively. Run EnergyArray.py first, because minima.py uses the array generated in EnergyArray.py. These files will be hard to generate for larger system sizes (>30). However, they're needed to analyze success rate for the simulated annealing, so other methods besides brute force may be necessary (spin glass server).

### Simulating Annealing
 The file MCAnnealing.py takes arguments

 ```
  -h, --help           show this help message and exit
  -N N                 Number of spins
  -C C                 J_Matrix confgiuration number
  -S S                 Annealing Rate in number of samples
  --Seed SEED          Random number seed
  --EQ EQ              Equilibration time for Monte Carlo Sampling
  --BStart BSTART      Starting temperature Beta
  --BEnd BEND          Ending temperature Beta
  --Constant CONSTANT  Constant Temperature Sampling? (True/False)
  ```

Running this code creates a .dat run-file of all of the states the monte carlo sampling reaches. It also creates a .dat stats-file of the minima found by the annealing. It creates a different set of files whether the parameter --Constant is set to true or not (This keeps a constant, minimum temperature throughout the sampling).


## Where files are stored

The file tree of where everything, inputs and outputs, are stored is a little complicated. FileTree.png on the repository is a picture of how the files are stored.

## Shell Scripts

MCAnnealingScript.sh and GenerateJsScript.sh are shell scripts to run the programs in their name with a variety of parameters.
