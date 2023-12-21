LOG_PREFIX="reproduce_src"

DATASETS="cifar10" # cifar10 or cifar100 or imagenet
METHODS="Src"

echo DATASETS: $DATASETS
echo METHODS: $METHODS

GPUS=(0 1 2 3 4 5 6 7) #available gpus
NUM_GPUS=${#GPUS[@]}

sleep 1 # prevent mistake
mkdir raw_logs # save console outputs here

#### Useful functions
wait_n() {
  #limit the max number of jobs as NUM_MAX_JOB and wait
  background=($(jobs -p))
  local default_num_jobs=8 #num concurrent jobs
  local num_max_jobs=${1:-$default_num_jobs}
  if ((${#background[@]} >= num_max_jobs)); then
    wait -n
  fi
}

###############################################################
##### Source Training; Source Evaluation: Source domains  #####
###############################################################
train_source_model() {
  i=0
  update_every_x="64"
  memory_size="64"
  for DATASET in $DATASETS; do
    for METHOD in $METHODS; do

      validation="--dummy"

      if [ "${DATASET}" = "cifar10" ]; then
        EPOCH=200
        MODEL="resnet18"
        TGT="test"
        SEEDS="0 1 2"
      elif [ "${DATASET}" = "cifar100" ]; then
        EPOCH=200
        MODEL="resnet18"
        TGT="test"
        SEEDS="0 1 2"
      elif [ "${DATASET}" = "imagenet" ]; then
        EPOCH=30
        MODEL="resnet18_pretrained"
        TGT="test"
        SEEDS="0" #for fair comparision with the pytorch-provided pre-trained model
      fi

      for SEED in $SEEDS ; do
        if [[ "$METHOD" == *"Src"* ]]; then
          for tgt in $TGT; do

            #### Train with BN
            if [ "${DATASET}" != "imagenet" ]; then
              python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method Src --tgt ${tgt} --model $MODEL --epoch $EPOCH --update_every_x ${update_every_x} --memory_size ${memory_size} --seed $SEED \
                --log_prefix ${LOG_PREFIX}_${SEED} \
                ${validation} \
                2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

              i=$((i + 1))
              wait_n
            fi

            ### Train with IABN
            for iabn_k in 4; do
              python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method Src --tgt ${tgt} --model $MODEL --epoch $EPOCH --update_every_x ${update_every_x} --memory_size ${memory_size} --seed $SEED \
                --iabn --iabn_k ${iabn_k} \
                --log_prefix ${LOG_PREFIX}_${SEED}_iabn_k${iabn_k} \
                ${validation} \
                2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

              i=$((i + 1))
              wait_n
            done
          done
        fi
      done
    done
  done

  wait
}

train_source_model
