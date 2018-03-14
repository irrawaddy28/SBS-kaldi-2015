#!/bin/bash -e

# Copyright 2015-2016  University of Illinois (Author: Amit Das)
# Apache 2.0
#

# Top most level script to train and test GMM-HMM and DNN for speech 
# recognition using probablistic transcripts (PT) generated 
# from crowdsource workers
#
# Try:
# 1) looping with increasing DT pseudo labels ... run for each language.
#  

. ./cmd.sh

TRAIN_LANG="AR CA HG MD UR" 
TEST_LANG="SW"
UNILANG_CODE=$(echo $TRAIN_LANG |sed 's/ /_/g')
stage=$1

dbn_dir=exp/dnn4_pretrain-dbn/${TEST_LANG}
dnn_dir=exp/dnn4_pretrain-dbn_dnn/${TEST_LANG}
hmm_dir=exp/tri3c/${TEST_LANG}
ali_dt_dir=exp/tri3c_ali/${TEST_LANG}
ali_pt_dir=exp/tri3cpt_ali/${TEST_LANG}
data_fmllr_dir=data-fmllr-tri3c/${TEST_LANG}

# =========================================
## Train a multilingual GMM-HMM system (exp/tri3b) using multilingual DT (deterministic transcripts) of training languages
## Test on a test language unseen during training
if [[ $stage -le 0 ]]; then
./run_hmm_multilingual.sh "${TRAIN_LANG}" "${TEST_LANG}"
fi
# =========================================

# =========================================
if [[ $stage -le 1 ]]; then
## Do a MAP adaptation of the multilingual GMM-HMM to PT (probabilistic transcripts) of test language. MAP adapted GMM-HMM is saved in exp/tri3c.
./run-pt-text-G-map-2.sh  "${TRAIN_LANG}" "${TEST_LANG}"
fi
# =========================================

# =========================================
if [[ $stage -le 2 ]]; then
## Train a multilingual DNN system using multilingual DT of training languages. Use tri3c_ali as targets.
## This nnet, trained using DT, is used to provide a good initialization of the shared hidden layers (SHLs) using DBN pre-training. If we start training
## with both PT + DT, the SHLs may be unreliable.
./run_dnn_adapt_to_multi_dt.sh "${TRAIN_LANG}" "${TEST_LANG}" ${hmm_dir} ${ali_dt_dir} ${data_fmllr_dir} ${dbn_dir}/indbn ${dnn_dir}/monosoftmax_dt
fi
# =========================================

# =========================================
## Train and test a monolingual GMM-HMM (exp/monolingual/tri3b)
## using monolingual DT (deterministic transcripts)
if [[ $stage -le 3 ]]; then
./run_hmm_monolingual.sh --stage 1 "${TEST_LANG}"
fi
# =========================================

# =========================================
## Train and test a monolingual DNN system (exp/monolingual/dnn4_pretrain-dbn_dnn)
## using monolingual DT (deterministic transcripts)
if [[ $stage -le 4 ]]; then
./run_dnn_monolingual.sh "${TEST_LANG}" exp/monolingual/tri3b/${TEST_LANG}  exp/monolingual/tri3b_ali/${TEST_LANG} \
                         exp/monolingual/data-fmllr-tri3b/${TEST_LANG} exp/monolingual/dnn4_pretrain-dbn/${TEST_LANG}/indbn \
                         exp/monolingual/dnn4_pretrain-dbn_dnn/${TEST_LANG}/monosoftmax_dt
fi
# =========================================

# =========================================
## Demonstrate the efficacy of a monosoftmax DNN trained with PTs where PTs are generated by an ASR system rather than crowdsource workers.
if [[ $stage -le 20 ]]; then 
  unsup_dir_tag="train"
  acwt=0.2
  feat_unsup_dir=data-fmllr-tri3b/${TEST_LANG}/${unsup_dir_tag}
  decoding_mdl_dir=${dnn_dir}/monosoftmax_dt # dnn mdl directory used to decode the unsup data
  lats_unsup_dir=${decoding_mdl_dir}/decode_${unsup_dir_tag}_text_G_${TEST_LANG} # dir where lattices generated by decoding unsup data will be saved
  
  # Now decode the training data using a reasonably well trained DNN model. The fMLLR transforms for train set are saved in tri3b_ali/decode_train
  # and the decoding lattices in the same DNN directory which is used for decoding the training data.
  # Note: We could also use GMM-HMM model tri3b/final.mdl for decoding but as of now the unsup lats scipt supports decoding using a nnet model.
  ./get_unsup_lats.sh --stage -2 --feats-nj 10 --unsup-dir-tag ${unsup_dir_tag} ${TEST_LANG} exp/tri3b_ali/${TEST_LANG} data-fmllr-tri3b/${TEST_LANG}/${TEST_LANG}/${unsup_dir_tag} \
     exp/tri3b/${TEST_LANG}/graph_text_G_${TEST_LANG}  ${decoding_mdl_dir}  ${lats_unsup_dir} || exit 1;   
   
  # Copy the fMLLR transforms for dev and eval sets from tri3b to tri3b_ali
  rm -rf exp/tri3b_ali/${TEST_LANG}/decode_dev_${TEST_LANG} exp/tri3b_ali/${TEST_LANG}/decode_eval_${TEST_LANG}
  cp -Lr exp/tri3b/${TEST_LANG}/decode_dev_${TEST_LANG} exp/tri3b_ali/${TEST_LANG}/decode_dev_${TEST_LANG}
  cp -Lr exp/tri3b/${TEST_LANG}/decode_eval_${TEST_LANG} exp/tri3b_ali/${TEST_LANG}/decode_eval_${TEST_LANG}
  
  # Now fine tune the DNN using the decoded unsup lattice. Use different levels of frame weighting derived from best path lattice.
  # Use the fMLLR transforms from tri3b_ali/decode_* and training lattices from the DNN directory ${lats_unsup_dir}
  thresh=0.5
  ./run_dnn_adapt_to_mono_pt_frame_wt.sh --transform-dir-train "exp/tri3b_ali/${TEST_LANG}/decode_${unsup_dir_tag}_${TEST_LANG}" --replace-softmax "true" --threshold ${thresh} \
	  "${TEST_LANG}" exp/tri3b_ali/${TEST_LANG}  ${lats_unsup_dir} \
	  ${dnn_dir}/monosoftmax_dt/final.nnet \
	  data-fmllr-tri3b/${TEST_LANG} ${dnn_dir}/monosoftmax_asrpt_fw${thresh}
  for thresh in 0.6 0.7 0.8 0.9 ; do
    (./run_dnn_adapt_to_mono_pt_frame_wt.sh --stage 2 --transform-dir-train "exp/tri3b_ali/${TEST_LANG}/decode_${unsup_dir_tag}_${TEST_LANG}" --replace-softmax "true" --threshold ${thresh} \
	  "${TEST_LANG}" exp/tri3b_ali/${TEST_LANG}  ${lats_unsup_dir} \
	  ${dnn_dir}/monosoftmax_dt/final.nnet \
	  data-fmllr-tri3b/${TEST_LANG} ${dnn_dir}/monosoftmax_asrpt_fw${thresh} || exit 1;) &
  done
  wait
fi
# =========================================

# =========================================
## Demonstrate the efficacy of a monosoftmax DNN trained with PTs where PTs are generated by crowdsource workers.
if [[ $stage -le 30 ]]; then
# Now, on top of the hidden layers of the multilingual DT system, create a new soft-max layer. This becomes a new DNN.
# Fine tune all layers of this new DNN using PT of the test language.
./run_dnn_adapt_to_mono_pt.sh --replace-softmax "true" "${TEST_LANG}" ${hmm_dir} ${ali_pt_dir} \
    ${dnn_dir}/monosoftmax_dt/final.nnet ${data_fmllr_dir} ${dnn_dir}/monosoftmax_pt || exit 1;
fi
if [[ $stage -le 31 ]]; then
# Now try the same thing using different levels of frame weighting derived from best path PT lattice. Do we get good improvements using frame weighting?
for thresh in 0.5 0.6 0.7 0.8 0.9 ; do
(./run_dnn_adapt_to_mono_pt_frame_wt.sh --stage 2 --replace-softmax "true" --threshold ${thresh} \
	"${TEST_LANG}" ${ali_pt_dir} ${ali_pt_dir}/decode_train \
	${dnn_dir}/monosoftmax_dt/final.nnet \
	${data_fmllr_dir} ${dnn_dir}/monosoftmax_pt_fw${thresh} || exit 1;) &
done
wait
fi
# =========================================

# =========================================
## Demonstrate the efficacy of a multisoftmax DNN trained with both PT and DT where softmax blocks are arranged as block 1:block2 = PT:DT. PT from crowdsource workers.
lats_pt_dir=${ali_pt_dir}/decode_train
if [[ $stage -le 40 ]]; then
# Train a 2 softmax DNN with PT and DT blocks. The configurations for such a DNN are
# a) SHLs are based on the earlier mono softmax DNN trained with multilingual DT
# b) use different levels of frame weighting thresholds derived from best path PT lattice, and
# c) create multiple copies of the PT data in the PT block

# Command for reference
#./run_dnn_multilingual.sh --dnn-init "exp/dnn4_pretrain-dbn_dnn/SW/monosoftmax_dt/final.nnet" --data-type-csl "pt:dt"  --lang-weight-csl "1.0:1.0"  \
#    --threshold-csl "${thresh}:0.0" --lat-dir-csl "$ptlatsdir:-" --dup-and-merge-csl "${num_copies}>>1:0>>0" \
#	"${TEST_LANG}:${UNILANG_CODE}" "$ptalidir:$dtalidir" \
#	"data-fmllr-tri3c/${TEST_LANG}/${TEST_LANG}/train:data-fmllr-tri3c/${TEST_LANG}/${UNILANG_CODE}/train" data-fmllr-tri3c/${TEST_LANG}/combined_fw${thresh}_cop${num_copies} \
#	exp/dnn4_pretrain-dbn_dnn/${TEST_LANG}/multisoftmax_pt_fw${thresh}_cop${num_copies} &
	
for thresh in 0.5 0.6 0.7 0.8 0.9 ; do
  for num_copies in 0 1 2 3 4; do
  ./run_dnn_multilingual.sh --dnn-init "${dnn_dir}/monosoftmax_dt/final.nnet" --data-type-csl "pt:dt"  --lang-weight-csl "1.0:1.0"  \
    --threshold-csl "${thresh}:0.0" --lat-dir-csl "${lats_pt_dir}:-" --dup-and-merge-csl "${num_copies}>>1:0>>0" \
	"${TEST_LANG}:${UNILANG_CODE}" "${ali_pt_dir}:${ali_dt_dir}" \
	"${data_fmllr_dir}/${TEST_LANG}/train:${data_fmllr_dir}/${UNILANG_CODE}/train" ${data_fmllr_dir}/combined_fw${thresh}_cop${num_copies} \
	$dnn_dir/multisoftmax_pt_fw${thresh}_cop${num_copies} &
  done
  wait
done
fi
# =========================================

# =========================================
## Demonstrate the efficacy of a multisoftmax DNN trained with PT, DT, and unsupervised data (PT:DT:Unsup).
nutts=4000
thresh=0.6
num_copies=0  
unsup_dir_tag="unsup_$nutts"
feat_unsup_dir=${data_fmllr_dir}/${TEST_LANG}/${unsup_dir_tag}
decoding_mdl_dir=${dnn_dir}/multisoftmax_pt_fw${thresh}_cop${num_copies}/decode_block_1_dev_text_G_${TEST_LANG} # dnn mdl directory used to decode the unsup data
lats_unsup_dir=${dnn_dir}/multisoftmax_pt_fw${thresh}_cop${num_copies}/decode_${unsup_dir_tag}_text_G_${TEST_LANG} # dir where lattices generated by decoding unsup data will be saved
if [[ $stage -le 50 ]]; then
  # Now decode the unsupervised data using a reasonably well trained DNN model  
  #./get_unsup_lats.sh --nutts ${nutts} --unsup-dir-tag ${unsup_dir_tag} ${TEST_LANG} ${ali_pt_dir} ${feat_unsup_dir} \
  #  ${hmm_dir}/graph_text_G_${TEST_LANG}  ${decoding_mdl_dir}  ${lats_unsup_dir} || exit 1;

feat_pt_dir=${data_fmllr_dir}/${TEST_LANG}/train
feat_dt_dir=${data_fmllr_dir}/${UNILANG_CODE}/train
# Train a 3 softmax DNN with PT,DT,UNSUP blocks. The configurations for such a DNN are
# a) SHLs are based on the earlier multisoftmax DNN trained with PT and DT
# b) use different levels of frame weighting threhsolds derived from best path PT lattice, and 
# c) create multiple copies of the PT/UNSUP data
# d) vary amount of UNSUP data

# Command for reference
#./run_dnn_multilingual.sh --dnn-init "exp/dnn4_pretrain-dbn_dnn/${TEST_LANG}/multisoftmax_pt_fw0.6_cop0/final.nnet" --remove_last_components 3 \
#                      --lang-weight-csl "1.0:1.0:1.0" --threshold-csl "0.7:0.0:0.8" \
#                      --lat-dir-csl "exp/dnn4_pretrain-dbn_dnn/${TEST_LANG}/multisoftmax_pt_fw0.6_cop0:-:exp/dnn4_pretrain-dbn_dnn/${TEST_LANG}/multisoftmax_pt_fw0.6_cop0/decode_unsup_4000_text_G_${TEST_LANG}" \
#                      --data-type-csl "pt:dt:unsup" --dup-and-merge-csl "4>>1:0>>2:1>>1" \
#                      "${TEST_LANG}:${UNILANG_CODE}:${TEST_LANG}" "exp/tri3cpt_ali/${TEST_LANG}:exp/tri3c_ali/${TEST_LANG}:exp/tri3cpt_ali/${TEST_LANG}"  \
#                      "data-fmllr-tri3c/${TEST_LANG}/${TEST_LANG}/train:data-fmllr-tri3c/${TEST_LANG}/${UNILANG_CODE}/train:data-fmllr-tri3c/${TEST_LANG}/${TEST_LANG}/unsup_4000" \
#                      "data-fmllr-tri3c/${TEST_LANG}/combined_fw0.70.00.8_cop401_unsup4000" \
#                      "exp/dnn4_pretrain-dbn_dnn/${TEST_LANG}/multisoftmax_pt_fw0.70.00.8_cop401_unsup4000"

for nutts_small_unsup in 4000 ; do  # 4000 3000 2000 1000
  for thresh_pt in 0.6 ; do  # 0.6 0.7 0.8
    for thresh_unsup in 0.9; do # 0.7 0.9    

      for num_copies_unsup in 0 1 2 3 4; do # 0 1
        for num_copies_dt in 0; do # 0 2 4 6
          for num_copies_pt in 0 2 4; do # 0 2 4
          
            num_copies=(${num_copies_pt} ${num_copies_dt} ${num_copies_unsup})
            thresh=(${thresh_pt} 0.0 ${thresh_unsup})
            nutts_small=${nutts_small_unsup}
           
            unsupsmall_dir_tag="unsup_${nutts_small}" 
            feat_unsupsmall_dir=${data_fmllr_dir}/${TEST_LANG}/${unsupsmall_dir_tag}
           
            if [ "$nutts_small" -lt "$nutts" ]; then             
              utils/subset_data_dir.sh ${feat_unsup_dir} ${nutts_small} ${feat_unsupsmall_dir}
            fi
           
            nnet_outdir=${dnn_dir}/multisoftmax_pt_fw${thresh[0]}${thresh[1]}${thresh[2]}_cop${num_copies[0]}${num_copies[1]}${num_copies[2]}_unsup${nutts_small}
            
            [ -f ${nnet_outdir}/final.nnet ] && echo "${nnet_outdir}/final.nnet exists. Skipping this run" && continue
           
            # To Do: Which dnn init does better? Does WER change significantly?
            # --dnn-init ${dnn_dir}/monosoftmax_dt/final.nnet --remove_last_components 2
            # --dnn-init ${decoding_mdl_dir}/final.nnet --remove_last_components 3
            ./run_dnn_multilingual.sh --dnn-init "${dnn_dir}/monosoftmax_dt/final.nnet" \
              --lang-weight-csl "1.0:1.0:1.0" --threshold-csl "${thresh[0]}:${thresh[1]}:${thresh[2]}" \
              --lat-dir-csl "${lats_pt_dir}:-:${lats_unsup_dir}" \
              --data-type-csl "pt:dt:unsup" --dup-and-merge-csl "${num_copies[0]}>>1:${num_copies[1]}>>2:${num_copies[2]}>>1" \
              "${TEST_LANG}:${UNILANG_CODE}:${TEST_LANG}" "${ali_pt_dir}:${ali_dt_dir}:${ali_pt_dir}" \
              "${feat_pt_dir}:${feat_dt_dir}:${feat_unsupsmall_dir}" \
              ${data_fmllr_dir}/combined_fw${thresh[0]}${thresh[1]}${thresh[2]}_cop${num_copies[0]}${num_copies[1]}${num_copies[2]}_unsup${nutts_small} \
              ${nnet_outdir} &
          done # num_copies_pt
          wait
        done # num_copies_dt
      done # num_copies_unsup
      
    done # thresh_unsup
  done # thresh_pt
done # nutts_small_unsup

fi
# =========================================