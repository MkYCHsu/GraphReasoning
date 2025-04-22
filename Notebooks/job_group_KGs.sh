#!/bin/bash

case_list=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
#case_list=(1 14 15 16 17)
for c in ${case_list[@]};
#c=0
#while [ $c -lt 4000 ]
do 
  sbatch job_KGs.sh ${c} 
  a=`expr $c + 1`
done


