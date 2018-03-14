#!/bin/bash

# Copyright 2012  Karel Vesely (Brno University of Technology)
# Apache 2.0

# Train neural network

# Begin configuration.

# training options
learn_rate=0.008
momentum=0
l1_penalty=0
l2_penalty=0
# data processing
minibatch_size=256
randomizer_size=32768
randomizer_seed=777
feature_transform=
# learn rate scheduling
max_iters=20
min_iters=0 # keep training, disable weight rejection, start learn-rate halving as usual,
keep_lr_iters=0 # fix learning rate for N initial epochs,
#start_halving_inc=0.5
#end_halving_inc=0.1
start_halving_impr=0.01
end_halving_impr=0.001
halving_factor=0.5
# Enable Teacher-Student (T-S) training
teacher_student=false  # false|true, if teacher_student=true, then enable T-S training
mlp_teacher=           # if teacher_student=true, then mlp_teacher must be set to a valid DNN
softmax_temperature=   # if teacher_student=true, then softmax-temperature must be set to a value > 0
rho_ts=                # if teacher_student=true, then rho must be set to a value [0, 1]
# Target interpolation
tgt_interp_mode="none" # "none|soft|hard"
rho=1.0   # any value in [0,1]
# misc.
use_gpu="wait"
verbose=1
# tool
train_tool="nnet-train-frmshuff"
frame_weights=
 
# End configuration.

echo "$0 $@"  # Print the command line for logging
[ -f path.sh ] && . ./path.sh; 

. parse_options.sh || exit 1;

if [ $# != 6 ]; then
   echo "Usage: $0 <mlp-init> <feats-tr> <feats-cv> <labels-tr> <labels-cv> <exp-dir>"
   echo " e.g.: $0 0.nnet scp:train.scp scp:cv.scp ark:labels_tr.ark ark:labels_cv.ark exp/dnn1"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>  # config containing options"
   echo "  --max-iters <N>         # number of training iterations"  
   exit 1;
fi

if $teacher_student; then
  # check if $mlp_teacher is not empty
  [[ -z $mlp_teacher ]] && echo "--mlp-teacher must be the path to an MLP" && exit 1
  
  # check if $softmax_temperature > 0
  if [[ -z $softmax_temperature || $( echo "$softmax_temperature <= 0" | bc ) == "1" ]]; then
    echo -e "Error: --softmax-temperature $softmax_temperature\nTemperature must be > 0"
    exit 1
  fi

  # check if $rho_ts lies in the interval [0,1]
  if [[ -z $rho_ts || $( echo "$rho_ts < 0" | bc ) == "1" || $( echo "$rho_ts > 1" | bc ) == "1" ]]; then
    echo -e "Error: --rho-ts $rho_ts\nrho must lie in [0,1] "
    exit 1
  fi
fi

mlp_init=$1
feats_tr=$2
feats_cv=$3
labels_tr=$4
labels_cv=$5
dir=$6

echo -e "
mlp_init=$1\n
feats_tr=$2\n
feats_cv=$3\n
labels_tr=$4\n
labels_cv=$5\n
dir=$6\n"

[ ! -d $dir ] && mkdir $dir
[ ! -d $dir/log ] && mkdir $dir/log
[ ! -d $dir/nnet ] && mkdir $dir/nnet

# Skip training
[ -e $dir/final.nnet ] && echo "'$dir/final.nnet' exists, skipping training" && exit 0

##############################
#start training

# choose mlp to start with
mlp_best=$mlp_init
mlp_base=${mlp_init##*/}; mlp_base=${mlp_base%.*}
# optionally resume training from the best epoch
[ -e $dir/.mlp_best ] && mlp_best=$(cat $dir/.mlp_best)
[ -e $dir/.learn_rate ] && learn_rate=$(cat $dir/.learn_rate)

echo -e "Training Parameters:\n
         mlp init = $mlp_best\n
         learn rate = $learn_rate\n
         max epochs = $max_iters\n
         minibatch size = $minibatch_size\n
         randomizer size = $randomizer_size\n
         frame weights  = $frame_weights\n
         momentum = $momentum\n
         l1-penalty = $l1_penalty\n
         l2-penalty = $l2_penalty\n
         teacher-student  = $teacher_student\n
         mlp_teacher = $mlp_teacher\n
         softmax-temperature = $softmax_temperature\n
         rho_ts =$rho_ts\n"

# Create teacher labels: Feedfwd feats through teacher n/w and save the posteriors as teacher labels
if $teacher_student; then
 nj=10
 postdir="$dir/ali-post"

 # For train.scp, create teacher labels in $postdir/post_teacher_tr.$n.ark, n = 1,..,$nj, and let $postdir/post_teacher_tr.scp point to these ark files
 split_scps=""
 for n in $(seq $nj); do
   split_scps="$split_scps $dir/train.${n}.scp"
 done
 utils/split_scp.pl $dir/train.scp $split_scps
 for n in $(seq $nj); do
  feats_split=$(echo $feats_tr|sed "s:train.scp:train.${n}.scp:g")
  nnet-forward --softmax-temperature=$softmax_temperature ${feature_transform:+ --feature-transform=$feature_transform} $mlp_teacher  "$feats_split" ark:- |\
    feat-to-post ark:- ark,scp:$postdir/post_teacher_tr.${n}.ark,$postdir/post_teacher_tr.${n}.scp || exit 1
 done
 cat $postdir/post_teacher_tr.*.scp > $postdir/post_teacher_tr.scp
 rm $split_scps $postdir/post_teacher_tr.*.scp
 labels_teacher_tr="scp:$postdir/post_teacher_tr.scp"

# Likewise for cv.scp
 split_scps=""
 for n in $(seq $nj); do
   split_scps="$split_scps $dir/cv.${n}.scp"
 done
 utils/split_scp.pl $dir/cv.scp $split_scps
 for n in $(seq $nj); do
  feats_split=$(echo $feats_cv|sed "s:cv.scp:cv.${n}.scp:g")
  nnet-forward --softmax-temperature=$softmax_temperature ${feature_transform:+ --feature-transform=$feature_transform} $mlp_teacher  "$feats_split" ark:- |\
    feat-to-post ark:- ark,scp:$postdir/post_teacher_cv.${n}.ark,$postdir/post_teacher_cv.${n}.scp || exit 1
 done
 cat $postdir/post_teacher_cv.*.scp > $postdir/post_teacher_cv.scp
 rm $split_scps $postdir/post_teacher_cv.*.scp
 labels_teacher_cv="scp:$postdir/post_teacher_cv.scp"

  #labels_teacher_tr="$dir/ali-post/post_teacher_tr.scp"
  #labels_teacher_cv="$dir/ali-post/post_teacher_cv.scp"
  #nnet-forward --softmax-temperature=$softmax_temperature ${feature_transform:+ --feature-transform=$feature_transform} $mlp_teacher  "$feats_tr" ark:- |\
  #  feat-to-post ark:- ark,scp:$dir/ali-post/post_teacher_tr.ark,$labels_teacher_tr
  #nnet-forward --softmax-temperature=$softmax_temperature ${feature_transform:+ --feature-transform=$feature_transform} $mlp_teacher  "$feats_cv" ark:- |\
  #  feat-to-post ark:- ark,scp:$dir/ali-post/post_teacher_cv.ark,$labels_teacher_cv
  #labels_teacher_tr="scp:$labels_teacher_tr"
  #labels_teacher_cv="scp:$labels_teacher_cv"
  
  #labels_teacher_tr="ark:nnet-forward --softmax-temperature=$softmax_temperature ${feature_transform:+ --feature-transform=$feature_transform} $mlp_teacher  \"$feats_tr\" ark:- | feat-to-post ark:- ark:- |"
  #labels_teacher_cv="ark:nnet-forward --softmax-temperature=$softmax_temperature ${feature_transform:+ --feature-transform=$feature_transform} $mlp_teacher  \"$feats_cv\" ark:- | feat-to-post ark:- ark:- |"
fi

# cross-validation on original network
log=$dir/log/iter00.initial.log; hostname>$log
if ! $teacher_student; then  
  $train_tool --cross-validate=true \
    --use-gpu=$use_gpu \
    --minibatch-size=$minibatch_size --randomizer-size=$randomizer_size --randomize=false --verbose=$verbose \
    --tgt-interp-mode=${tgt_interp_mode} --rho=${rho} \
    ${feature_transform:+ --feature-transform=$feature_transform} \
    ${frame_weights:+ "--frame-weights=$frame_weights"} \
    "$feats_cv" "$labels_cv" $mlp_best \
    2>> $log || exit 1;
else  
  $train_tool --cross-validate=true \
    --use-gpu=$use_gpu \
    --minibatch-size=$minibatch_size --randomizer-size=$randomizer_size --randomize=false --verbose=$verbose \
    --softmax-temperature=${softmax_temperature}  --rho=${rho_ts} \
    ${feature_transform:+ --feature-transform=$feature_transform} \
    ${frame_weights:+ "--frame-weights=$frame_weights"} \
    "$feats_cv" "$labels_cv" "$labels_teacher_cv"  $mlp_best \
    2>> $log | tee || exit 1;
fi


loss_prev=$(cat $dir/log/iter00.initial.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
loss_type=$(cat $dir/log/iter00.initial.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $5; }')
echo "CROSSVAL PRERUN AVG.LOSS $(printf "%.4f" $loss_prev) $loss_type"

# resume lr-halving
halving=0
[ -e $dir/.halving ] && halving=$(cat $dir/.halving)
# training
for iter in $(seq -w $max_iters); do
  echo -n "ITERATION $iter: "
  mlp_next=$dir/nnet/${mlp_base}_iter${iter}
  
  # skip iteration if already done
  [ -e $dir/.done_iter$iter ] && echo -n "skipping... " && ls $mlp_next* && continue 
  
  # training
  log=$dir/log/iter${iter}.tr.log; hostname>$log

  if ! $teacher_student; then
    $train_tool \
      --use-gpu=$use_gpu \
      --learn-rate=$learn_rate --momentum=$momentum --l1-penalty=$l1_penalty --l2-penalty=$l2_penalty \
      --minibatch-size=$minibatch_size --randomizer-size=$randomizer_size --randomize=true --verbose=$verbose \
      --tgt-interp-mode=${tgt_interp_mode} --rho=${rho} \
      --binary=true \
      ${feature_transform:+ --feature-transform=$feature_transform} \
      ${frame_weights:+ "--frame-weights=$frame_weights"} \
      ${randomizer_seed:+ --randomizer-seed=$randomizer_seed} \
      "$feats_tr" "$labels_tr" $mlp_best $mlp_next \
      2>> $log || exit 1;     
  else
    $train_tool \
      --use-gpu=$use_gpu \
      --learn-rate=$learn_rate --momentum=$momentum --l1-penalty=$l1_penalty --l2-penalty=$l2_penalty \
      --minibatch-size=$minibatch_size --randomizer-size=$randomizer_size --randomize=true --verbose=$verbose \
      --softmax-temperature=${softmax_temperature}  --rho=${rho_ts} \
      --binary=true \
      ${feature_transform:+ --feature-transform=$feature_transform} \
      ${frame_weights:+ "--frame-weights=$frame_weights"} \
      ${randomizer_seed:+ --randomizer-seed=$randomizer_seed} \
      "$feats_tr" "$labels_tr" "$labels_teacher_tr" $mlp_best $mlp_next \
      2>> $log | tee || exit 1;
  fi

  tr_loss=$(cat $dir/log/iter${iter}.tr.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
  tr_fr_acc=$(cat $dir/log/iter${iter}.tr.log | grep FRAME_ACCURACY | awk '{print $3" (task "NR")"}'| tr '\n' ' '|sed 's/,$//')
  echo -n "TRAIN AVG.LOSS $(printf "%.4f" $tr_loss), (lrate$(printf "%.6g" $learn_rate)), FRAME ACC $tr_fr_acc, "
  
  # cross-validation
  log=$dir/log/iter${iter}.cv.log; hostname>$log
  if ! $teacher_student; then    
    $train_tool --cross-validate=true \
      --use-gpu=$use_gpu \
      --minibatch-size=$minibatch_size --randomizer-size=$randomizer_size --randomize=false --verbose=$verbose \
      --tgt-interp-mode=${tgt_interp_mode} --rho=${rho} \
      ${feature_transform:+ --feature-transform=$feature_transform} \
      ${frame_weights:+ "--frame-weights=$frame_weights"} \
     "$feats_cv" "$labels_cv" $mlp_next \
     2>>$log || exit 1;
   else
    $train_tool --cross-validate=true \
      --use-gpu=$use_gpu \
      --minibatch-size=$minibatch_size --randomizer-size=$randomizer_size --randomize=false --verbose=$verbose \
      --softmax-temperature=${softmax_temperature}  --rho=${rho_ts} \
      ${feature_transform:+ --feature-transform=$feature_transform} \
      ${frame_weights:+ "--frame-weights=$frame_weights"} \
      "$feats_cv" "$labels_cv" "$labels_teacher_cv"  $mlp_next \
      2>> $log | tee || exit 1;
  fi

  loss_new=$(cat $dir/log/iter${iter}.cv.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
  cv_fr_acc=$(cat $dir/log/iter${iter}.cv.log | grep FRAME_ACCURACY | awk '{print $3" (task "NR")"}'| tr '\n' ' '|sed 's/,$//')
  echo -n "CROSSVAL AVG.LOSS $(printf "%.4f" $loss_new), FRAME ACC $cv_fr_acc, "
  
  # accept or reject new parameters (based on objective function)
  # ( Use ${iter#0} instead of ${iter} since when iter takes 08, bash thinks it is octal 08 but in octal system, 08 doesn't exist)
  if [[ "1" == $(echo "$loss_new < $loss_prev" | bc) || ${iter#0} -le ${keep_lr_iters#0} || ${iter#0} -le ${min_iters#0} ]]; then
    mlp_best=$dir/nnet/${mlp_base}_iter${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_loss)_cv$(printf "%.4f" $loss_new)
    [ $iter -le $min_iters ] && mlp_best=${mlp_best}_min-iters-$min_iters
    [ $iter -le $keep_lr_iters ] && mlp_best=${mlp_best}_keep-lr-iters-$keep_lr_iters
    mv $mlp_next $mlp_best
    nnet_status="nnet accepted ($(basename $mlp_best))"
    echo $mlp_best > $dir/.mlp_best 
  else
    mlp_reject=$dir/nnet/${mlp_base}_iter${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_loss)_cv$(printf "%.4f" $loss_new)_rejected
    mv $mlp_next $mlp_reject
    nnet_status="nnet rejected ($(basename $mlp_reject))"
  fi

  # create .done file as a mark that iteration is over
  touch $dir/.done_iter$iter
  
  # no learn-rate halving yet, if keep_lr_iters set accordingly
  [ $iter -le $keep_lr_iters ] && continue

  # stopping criterion  
  rel_impr=$(bc <<< "scale=10; ($loss_prev-$loss_new)/$loss_prev")
  echo "rel impr = $(printf "%.5f" $rel_impr), $nnet_status, `date`"  
  if [[ 1 == $halving && 1 == $(bc <<< "$rel_impr > 0") && 1 == $(bc <<< "$rel_impr < $end_halving_impr")  ]]; then
    if [ $iter -le $min_iters ]; then
      echo "we were supposed to finish, but we continue as min_iters : $min_iters "
      continue
    else
      echo "finished, too small rel. improvement $rel_impr, threshold $end_halving_impr "
      break
    fi
  fi

  # start annealing when improvement is low  
  if [[ 1 == $(bc <<< "$rel_impr < $start_halving_impr") ]]; then
    halving=1
    echo $halving >$dir/.halving
  fi
  
  # do annealing
  if [ 1 == $halving ]; then
    learn_rate=$(awk "BEGIN{print($learn_rate*$halving_factor)}")
    echo $learn_rate >$dir/.learn_rate
  fi

  # if loss_new <= loss_prev, then loss_prev  =  loss_new 
  if [[ "1" == $(echo "$loss_new < $loss_prev" | bc) || ${iter#0} -le ${keep_lr_iters#0} || ${iter#0} -le ${min_iters#0} ]]; then
    loss_prev=$loss_new
  fi
done


# select the best network
if [ $mlp_best != $mlp_init ]; then 
  mlp_final=${mlp_best}_final_
  ( cd $dir/nnet; ln -s $(basename $mlp_best) $(basename $mlp_final); )
  ( cd $dir; ln -s nnet/$(basename $mlp_final) final.nnet; )
  echo "Succeeded training the Neural Network : $dir/final.nnet"
else
  echo "Error training neural network..."
  exit 1
fi

# delete the big ark files
if $teacher_student; then
  rm -rf $dir/ali-post/post_teacher*.ark
fi
