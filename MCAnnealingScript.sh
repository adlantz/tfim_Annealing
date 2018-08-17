#!/bin/bash

k=$SLURM_ARRAY_TASK_ID

for j in {1..15}
do
  for i in {1..1000}
    do
      python ./MCAnnealing.py --Seed $i -S $((2**$j)) -C $k
      python ./MCAnnealing.py --Seed $i -S $((2**$j)) --Constant True -C$k
    done
  done
