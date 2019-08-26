#!/bin/bash


frame_time=0

cc=0
while IFS=':' read -r line || [[ -n $line ]]; do
    objdet_time[cc]=`echo $line | awk -F: '{print $2}'`
  	cc=$(($cc+1))
done < frame_timing.txt

time=0
for ((c=1; c < cc; c++)); 
do
time=`bc -l <<< $time+${objdet_time[$c]}`
done
frame_time=`bc -l <<< $frame_time+$time`

echo "Average Time: "`bc -l <<< $time/$c` > ave_timing.txt
echo "Average FPS: "`bc -l <<< $c/$time` >> ave_timing.txt

