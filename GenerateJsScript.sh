#!/bin/bash

for i in {101..200}
do
  python GenerateJs.py -Seed $i
done
