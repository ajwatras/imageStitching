#!/bin/bash

cc=0
while IFS=':' read -r line || [[ -n $line ]]; do
    objdet_time[cc]=`echo $line | awk -F: '{print $2}'`
  	cc=$(($cc+1))
done < obj_det_timing.txt

time=0
for ((c=1; c < cc; c++)); 
do
time=`bc -l <<< $time+${objdet_time[$c]}`
done

echo "Average Detection Time: "`bc -l <<< $time/$c` > ave_timing.txt
echo "Average Detection FPS: "`bc -l <<< $c/$time` >> ave_timing.txt

cc=0
while IFS=':' read -r line || [[ -n $line ]]; do
    objdet_time[cc]=`echo $line | awk -F: '{print $2}'`
  	cc=$(($cc+1))
done < obj_align_timing.txt

time=0
for ((c=1; c < cc; c++)); 
do
time=`bc -l <<< $time+${objdet_time[$c]}`
done

echo "Average Alignment Time: "`bc -l <<< $time/$c` >> ave_timing.txt
echo "Average Alignment FPS: "`bc -l <<< $c/$time` >> ave_timing.txt

cc=0
while IFS=':' read -r line || [[ -n $line ]]; do
    objdet_time[cc]=`echo $line | awk -F: '{print $2}'`
  	cc=$(($cc+1))
done < obj_warp_timing.txt

time=0
for ((c=1; c < cc; c++)); 
do
time=`bc -l <<< $time+${objdet_time[$c]}`
done

echo "Average Warping Time: "`bc -l <<< $time/$c` >> ave_timing.txt
echo "Average Warping FPS: "`bc -l <<< $c/$time` >> ave_timing.txt