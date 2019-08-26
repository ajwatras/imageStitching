#!/bin/bash


frame_time=0

cc=0
while IFS=':' read -r line || [[ -n $line ]]; do
    step_time[$cc]=`echo -n $line | awk -F: '{print $2}'`
  	cc=$(($cc+1))
done < $1

time=0
for ((c=1; c < cc; c++)); 
do
time=`bc -l <<< $time+${step_time[$c]}`
done



echo $1
echo "Average Time: "`bc -l <<< $time/$c` 
echo "Average FPS: "`bc -l <<< $c/$time`
