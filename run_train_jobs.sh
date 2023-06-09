#!/bin/bash
set -e

GPU=0
num_gpus=4

# Loop through model types
for model in LSTM LTC
do

  # Loop through training mode
  for train_mode in adversarial nonadversarial
  do

    # Loop through testing mode
    for test_mode in adversarial nonadversarial
    do

      # Loop through sampling mode
      for sampling in regular irregular
      do

       echo $model $train_mode $test_mode $sampling
       savedir="logs/${model}_${train_mode}_${test_mode}_${sampling}"
       mkdir -p $savedir
       CUDA_VISIBLE_DEVICES=$GPU python train.py --model_type $model --train $train_mode --test $test_mode --sample $sampling > ${savedir}/log 2>&1 &

       # Increment GPU so we parallelize experiments on different gpus
       GPU=$((GPU + 1))
       GPU=$((GPU % num_gpus))
       
    done
  done
 done
done
