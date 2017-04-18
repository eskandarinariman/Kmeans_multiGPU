#!/bin/bash

set -e

data="./data/data"
rank0="ug58.eecg.utoronto.ca"
hosts=(58 59 60 61)
sizes=(8 16)
iters=(2 4 8 16 32)
runs=(0 1 2 3)
types=("mpi" "streaming")


for size in ${sizes[@]}; do
	for iter in ${iters[@]}; do
		for run in ${runs[@]}; do
			for type in ${types[@]}; do

				for i in {0..3}; do
					if ! ssh $rank0 stat "/tmp/data$i" \> /dev/null 2\>\&1; then
						echo "missing data$i"
						scp "$data$i" "ug58.eecg.utoronto.ca:/tmp/data$i"
					fi
				done

				echo "type $type run $run iters $iter size $size"

				if [[ $type == "mpi" ]]; then
					ssh $rank0 << EOF > "./logs/${type}_${size}_${iter}_${run}.log"
					cd ece1782/project/Kmeans_multiGPU/cuda_aware_mpi; 
					export LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64:"
					/guest/l/liondavi/ece1782/project/mpi/bin/mpiexec -x LD_LIBRARY_PATH -rf ug$size.mpi coalesce_gpu_sum_remote /tmp/data0 3800000 256 128 $iter
EOF
				else
					if [[ "$size" == "8" ]]; then
						ssh $rank0 << EOF > "./logs/${type}_${size}_${iter}_${run}.log"
						cd ece1782/project/Kmeans_multiGPU/cuda_streaming;
						export LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64:"
						./coalesce_gpu_sum_pin 3800000 256 128 $iter 4 /tmp/data0 /tmp/data1
EOF
					else
						ssh $rank0 << EOF > "./logs/${type}_${size}_${iter}_${run}.log"
						cd ece1782/project/Kmeans_multiGPU/cuda_streaming;
						export LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64:"
						./coalesce_gpu_sum_pin 3800000 256 128 $iter 4 /tmp/data0 /tmp/data1 /tmp/data2 /tmp/data3
EOF
					fi
				fi

			done
		done
	done
done

