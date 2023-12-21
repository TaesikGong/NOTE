
SRC_PREFIX="reproduce_src"
LOG_PREFIX="reproduce_methods"



DATASETS="cifar10" # cifar10 or cifar100 or imagenet
METHODS="NOTE" #Src BN_Stats ONDA PseudoLabel TENT CoTTA NOTE NOTE_iid

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


test_time_adaptation() {
  ###############################################################
  ###### Run Baselines & NOTE; Evaluation: Target domains  ######
  ###############################################################

  i=0

  for DATASET in $DATASETS; do
    for METHOD in $METHODS; do

      update_every_x="64"
      memory_size="64"
      SEED="0"
      lr="0.001" #other baselines
      validation="--dummy"
      weight_decay="0"
      if [ "${DATASET}" = "cifar10" ]; then
        MODEL="resnet18"
        CP_base="log/cifar10/Src/tgt_test/"${SRC_PREFIX}

        #              TGTS="test"
        TGTS="gaussian_noise-5
            shot_noise-5
            impulse_noise-5
            defocus_blur-5
            glass_blur-5
            motion_blur-5
            zoom_blur-5
            snow-5
            frost-5
            fog-5
            brightness-5
            contrast-5
            elastic_transform-5
            pixelate-5
            jpeg_compression-5"

      elif [ "${DATASET}" = "cifar100" ]; then
        MODEL="resnet18"
        CP_base="log/cifar100/Src/tgt_test/"${SRC_PREFIX}
        #              TGTS="test"
        TGTS="gaussian_noise-5
            shot_noise-5
            impulse_noise-5
            defocus_blur-5
            glass_blur-5
            motion_blur-5
            zoom_blur-5
            snow-5
            frost-5
            fog-5
            brightness-5
            contrast-5
            elastic_transform-5
            pixelate-5
            jpeg_compression-5"
      elif [ "${DATASET}" = "imagenet" ]; then
        MODEL="resnet18_pretrained"
        CP_base="log/imagenet/Src/tgt_test/"${SRC_PREFIX}
        TGTS="gaussian_noise-5
            shot_noise-5
            impulse_noise-5
            defocus_blur-5
            glass_blur-5
            motion_blur-5
            zoom_blur-5
            snow-5
            frost-5
            fog-5
            brightness-5
            contrast-5
            elastic_transform-5
            pixelate-5
            jpeg_compression-5"
      fi

      for SEED in 0 1 2; do #multiple seeds
          if [ "${METHOD}" = "Src" ]; then
            EPOCH=0
            if [ "${DATASET}" = "imagenet" ]; then
              CP="" # use pytorch-provided model
            else
              CP="--load_checkpoint_path "${CP_base}_${SEED}/cp/cp_last.pth.tar
            fi
            for TGT in $TGTS; do
            python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH --update_every_x ${update_every_x} ${CP} --seed $SEED \
              --log_prefix ${LOG_PREFIX}_${SEED} \
              ${validation} \
              2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

            i=$((i + 1))
            wait_n
          done


        elif [ "${METHOD}" = "NOTE" ]; then

          lr="0.0001"
          EPOCH=1
          memory_type="PBRS"
          loss_scaler=0
          iabn_k=4
          bn_momentum=0.01

          if [ "${DATASET}" = "imagenet" ]; then
            CP="--load_checkpoint_path "${CP_base}_0_iabn_k${iabn_k}/cp/cp_last.pth.tar
          else
            CP="--load_checkpoint_path "${CP_base}_${SEED}_iabn_k${iabn_k}/cp/cp_last.pth.tar
          fi

          for dist in 0 1; do #dist 0: non-i.i.d. / dist 1: i.i.d.
            for TGT in $TGTS; do
              python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH ${CP} --seed $SEED \
                --remove_cp --online --use_learned_stats --lr ${lr} --weight_decay ${weight_decay} --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} --memory_type ${memory_type} --bn_momentum ${bn_momentum} \
                --iabn --iabn_k ${iabn_k} \
                --log_prefix ${LOG_PREFIX}_${SEED}_dist${dist}_iabn_k${iabn_k}_mt${bn_momentum} \
                --loss_scaler ${loss_scaler} \
                ${validation} \
                2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

              i=$((i + 1))
              wait_n
            done
          done
        elif [ "${METHOD}" = "NOTE_iid" ]; then # NOTE given an i.i.d. assumption

          lr="0.001"
          EPOCH=1
          memory_type="FIFO"
          loss_scaler=0
          iabn_k=4

          if [ "${DATASET}" = "imagenet" ]; then
            CP="--load_checkpoint_path "${CP_base}_0_iabn_k${iabn_k}/cp/cp_last.pth.tar
          else
            CP="--load_checkpoint_path "${CP_base}_${SEED}_iabn_k${iabn_k}/cp/cp_last.pth.tar
          fi
          for dist in 1; do # dist 1: i.i.d.
            ### Train with IABN

            for TGT in $TGTS; do
              python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method "NOTE" --tgt ${TGT} --model $MODEL --epoch $EPOCH ${CP} --seed $SEED \
                --remove_cp --online --lr ${lr} --weight_decay ${weight_decay} --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} --memory_type ${memory_type} \
                --iabn --iabn_k ${iabn_k} \
                --log_prefix ${LOG_PREFIX}_iid_${SEED}_dist${dist}_iabn_k${iabn_k} \
                --loss_scaler ${loss_scaler} \
                ${validation} \
                2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

              i=$((i + 1))
              wait_n
            done
          done
        elif [ "${METHOD}" = "BN_Stats" ]; then

          if [ "${DATASET}" = "imagenet" ]; then
            CP="" # use pytorch-provided model
          else
            CP="--load_checkpoint_path "${CP_base}_${SEED}/cp/cp_last.pth.tar
          fi
          #### Train with BN
          for dist in 0 1; do
            for TGT in $TGTS; do

              python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH ${CP} --seed $SEED \
                --remove_cp --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} \
                --log_prefix ${LOG_PREFIX}_${SEED}_dist${dist} \
                ${validation} \
                2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

              i=$((i + 1))
              wait_n
            done

          done
        elif [ "${METHOD}" = "ONDA" ]; then

          EPOCH=1
          #### Train with BN
          update_every_x=10
          memory_size=10
          bn_momentum=0.1
          if [ "${DATASET}" = "imagenet" ]; then
            CP="" # use pytorch-provided model
          else
            CP="--load_checkpoint_path "${CP_base}_${SEED}/cp/cp_last.pth.tar
          fi
          for dist in 0 1; do
            for TGT in $TGTS; do
              python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH ${CP} --seed $SEED \
                --remove_cp --online --use_learned_stats --weight_decay ${weight_decay} --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} \
                --bn_momentum ${bn_momentum} \
                --log_prefix ${LOG_PREFIX}_${SEED}_dist${dist} \
                ${validation} \
                2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

              i=$((i + 1))
              wait_n
            done
          done
        elif [ "${METHOD}" = "PseudoLabel" ]; then
          EPOCH=1
          lr=0.001
          if [ "${DATASET}" = "imagenet" ]; then
            CP="" # use pytorch-provided model
          else
            CP="--load_checkpoint_path "${CP_base}_${SEED}/cp/cp_last.pth.tar
          fi
          for dist in 0 1; do
            for TGT in $TGTS; do

              python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH ${CP} --seed $SEED \
                --remove_cp --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} \
                --lr ${lr} --weight_decay ${weight_decay} \
                --log_prefix ${LOG_PREFIX}_${SEED}_dist${dist} \
                ${validation} \
                2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

              i=$((i + 1))
              wait_n
            done
          done
        elif [ "${METHOD}" = "TENT" ]; then
          EPOCH=1
          if [ "${DATASET}" = "imagenet" ]; then
            CP="" # use pytorch-provided model
            lr=0.00025 #referred to the paper
          else
            lr=0.001
            CP="--load_checkpoint_path "${CP_base}_${SEED}/cp/cp_last.pth.tar
          fi
          #### Train with BN
          for dist in 0 1; do
            for TGT in $TGTS; do

              python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH ${CP} --seed $SEED \
                --remove_cp --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} \
                --lr ${lr} --weight_decay ${weight_decay} \
                --log_prefix ${LOG_PREFIX}_${SEED}_dist${dist} \
                ${validation} \
                2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

              i=$((i + 1))
              wait_n
            done
          done
        elif [ "${METHOD}" = "LAME" ]; then
          EPOCH=1
          #### Train with BN
          if [ "${DATASET}" = "imagenet" ]; then
            CP="" # use pytorch-provided model
          else
            CP="--load_checkpoint_path "${CP_base}_${SEED}/cp/cp_last.pth.tar
          fi
          for dist in 0 1; do
            for TGT in $TGTS; do

              python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH ${CP} --seed $SEED \
                --remove_cp --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} \
                --lr ${lr} --weight_decay ${weight_decay} \
                --log_prefix ${LOG_PREFIX}_${SEED}_dist${dist} \
                ${validation} \
                2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

              i=$((i + 1))
              wait_n
            done
          done
        elif [ "${METHOD}" = "CoTTA" ]; then
          lr=0.001
          EPOCH=1

          if [ "${DATASET}" = "cifar10" ]; then
            aug_threshold=0.92 #value reported from the official code
          elif [ "${DATASET}" = "cifar100" ]; then
            aug_threshold=0.72 #value reported from the official code
          elif [ "${DATASET}" = "imagenet" ]; then
            aug_threshold=0.1 #value reported from the official code
          fi

          if [ "${DATASET}" = "imagenet" ]; then
            CP="" # use pytorch-provided model
          else
            CP="--load_checkpoint_path "${CP_base}_${SEED}/cp/cp_last.pth.tar
          fi
          for dist in 0 1; do
            for TGT in $TGTS; do

              python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset $DATASET --method ${METHOD} --tgt ${TGT} --model $MODEL --epoch $EPOCH ${CP} --seed $SEED \
                --remove_cp --online --tgt_train_dist ${dist} --update_every_x ${update_every_x} --memory_size ${memory_size} \
                --lr ${lr} --weight_decay ${weight_decay} \
                --aug_threshold ${aug_threshold} \
                --log_prefix ${LOG_PREFIX}_${SEED}_dist${dist} \
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

test_time_adaptation