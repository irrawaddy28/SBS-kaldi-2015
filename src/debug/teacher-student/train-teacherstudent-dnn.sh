#!/bin/bash

. ./path.sh
utils=../../../egs/wsj/s5/utils

nnet_teacher=common/nnet.teacher
feats=common/feats.txt

# ----------------- 1-layer dnn (affine xform + softmax)  -----------------
nnet_proto=exp/dnn/proto1L  # student proto
nnet_init=exp/dnn/nnet1L.in # student nnet
nnet_out=exp/dnn/nnet1L.out # output of student nnet
post1=exp/dnn/post1.txt
post2=exp/dnn/post2.txt
T=2.0
rho=0.2

# Create the student DNN
python $utils/nnet/make_nnet_proto.py 5 3 0 1000 > $nnet_proto
nnet-initialize --binary=false $nnet_proto $nnet_init

echo "Features"
cat $feats

echo "Nnet"
cat $nnet_init

# Feedforward the features through the teacher network using a temperature based softmax to get soft labels
nnet-forward --softmax-temperature=$T $nnet_teacher ark:$feats ark,t:-| feat-to-post ark:- ark,t:$post2

# Train the student DNN with Teacher-Student loss
nnet-train-frmshuff --verbose=4 --randomize=false --use-gpu=no --objective-function=ts --minibatch-size=6 --softmax-temperature=$T --rho=$rho \
  ark:$feats ark:$post1 ark:$post2 $nnet_init $nnet_out

# Expected Output (1 layer, T = 2, rho  = 0.2)
# nnet-initialize --binary=false exp/dnn/proto1L exp/dnn/nnet1L.in 
# VLOG[1] (nnet-initialize:Init():nnet-nnet.cc:378) <NnetProto>
# VLOG[1] (nnet-initialize:Init():nnet-nnet.cc:378) <AffineTransform> <InputDim> 5 <OutputDim> 3 <BiasMean> 0.000000 <BiasRange> 0.000000 <ParamStddev> 1.750000
# VLOG[1] (nnet-initialize:Init():nnet-nnet.cc:378) <Softmax> <InputDim> 3 <OutputDim> 3
# VLOG[1] (nnet-initialize:Init():nnet-nnet.cc:378) </NnetProto>
# LOG (nnet-initialize:main():nnet-initialize.cc:64) Written initialized model to exp/dnn/nnet1L.in
# Features
# s1u1 [ 
#    -9.2396   -7.4065    -13.1945  -8.7807   -9.5492 
#   -11.8494  -10.6132   -9.5155   -4.9394   -6.0834 
#   -1.9090    4.2920     1.0259   -0.0892    1.0108 
#   -0.2899   -0.1756     2.1068    1.9927    2.0042 
#   10.9496    8.2924    11.0143   12.3055   10.6914 
#   11.4633   11.0280     9.5709   10.4156    8.8866 ]Nnet
# <Nnet> 
# <AffineTransform> 3 5 
# <LearnRateCoef> 1 <BiasLearnRateCoef> 1 <MaxNorm> 0  [
#   2.068715 -0.514695 -2.912265 0.08992745 0.05172325 
#   0.07870442 -0.449972 -0.4911808 -0.5379869 -0.3142128 
#   2.649026 -1.479693 -0.5505048 2.179074 0.1373862 ]
#  [ 0 0 0 ]
# <Softmax> 3 3 
# </Nnet> 
# feat-to-post ark:- ark,t:exp/dnn/post2.txt 
# nnet-forward --softmax-temperature=2.0 common/nnet.teacher ark:common/feats.txt ark,t:- 
# LOG (nnet-forward:SelectGpuId():cu-device.cc:83) Manually selected to compute on CPU.
# LOG (nnet-forward:main():nnet-forward.cc:221) Done 1 files in 2.08219e-06min, (fps 48026.4)
# LOG (feat-to-post:main():feat-to-post.cc:71) Converted 1 alignments.
# nnet-train-frmshuff --verbose=4 --randomize=false --use-gpu=no --objective-function=ts --minibatch-size=6 --softmax-temperature=2.0 --rho=0.2 ark:common/feats.txt ark:exp/dnn/post1.txt ark:exp/dnn/post2.txt exp/dnn/nnet1L.in exp/dnn/nnet1L.out 
# LOG (nnet-train-frmshuff:SelectGpuId():cu-device.cc:83) Manually selected to compute on CPU.
# LOG (nnet-train-frmshuff:Init():nnet-randomizer.cc:31) Seeding by srand with : 777
# LOG (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:190) Objective Function = ts

# LOG (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:193) TRAINING STARTED
# VLOG[3] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:205) Reading s1u1
# LOG (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:273) Finished filling randomizer. num done = 1, num_no_tgt_mat = 0, num_no_frame_wts = 0

# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:295) Mini-batch loop

# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:300) nnet_in =  [
#   -9.2396 -7.4065 -13.1945 -8.7807 -9.5492 
#   -11.8494 -10.6132 -9.5155 -4.9394 -6.0834 
#   -1.909 4.292 1.0259 -0.0892 1.0108 
#   -0.2899 -0.1756 2.1068 1.9927 2.0042 
#   10.9496 8.2924 11.0143 12.3055 10.6914 
#   11.4633 11.028 9.5709 10.4156 8.8866 ]


# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:301) nnet_out =  [
#   0.993501 0.00649929 8.25932e-22 
#   0.00557794 0.994422 5.49713e-16 
#   0.00193302 0.997963 0.000103588 
#   9.0029e-05 0.00357513 0.996335 
#   7.17391e-23 1.41891e-25 1 
#   1.36494e-18 2.23234e-22 1 ]


# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:323) nnet_in =  [
#   -9.2396 -7.4065 -13.1945 -8.7807 -9.5492 
#   -11.8494 -10.6132 -9.5155 -4.9394 -6.0834 
#   -1.909 4.292 1.0259 -0.0892 1.0108 
#   -0.2899 -0.1756 2.1068 1.9927 2.0042 
#   10.9496 8.2924 11.0143 12.3055 10.6914 
#   11.4633 11.028 9.5709 10.4156 8.8866 ]


# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:324) Temperature of nnet1 =  
#   Temperature = 1

# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:325) Temperature of nnet2 =  
#   Temperature = 2

# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:458) nnet_out1 =  [
#   0.993501 0.00649929 8.25932e-22 
#   0.00557794 0.994422 5.49713e-16 
#   0.00193302 0.997963 0.000103588 
#   9.0029e-05 0.00357513 0.996335 
#   7.17391e-23 1.41891e-25 1 
#   1.36494e-18 2.23234e-22 1 ]


# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:459) nnet_tgt =  [
#   1 0 0 
#   1 0 0 
#   0 1 0 
#   0 1 0 
#   0 0 1 
#   0 0 1 ]


# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:460) nnet_out2 =  [
#   0.925171 0.0748292 2.66753e-11 
#   0.0696764 0.930324 2.18734e-08 
#   0.0417482 0.948587 0.00966439 
#   0.00888884 0.0560144 0.935097 
#   8.46989e-12 3.76684e-13 1 
#   1.16831e-09 1.4941e-11 1 ]


# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:461) nnet_tgt2 =  [
#   0.996668 0.000423341 0.00290883 
#   0.995427 1.42847e-09 0.00457299 
#   3.54915e-05 0.999949 1.59372e-05 
#   0.367255 0.0195188 0.613226 
#   0.000484859 0.18416 0.815355 
#   2.62384e-07 0.999678 0.000321986 ]


# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:402) virtual void kaldi::nnet1::TS::Eval2(const kaldi::VectorBase<float>&, const kaldi::CuMatrixBase<float>&, const kaldi::CuMatrixBase<float>&, const kaldi::CuMatrixBase<float>&, const kaldi::CuMatrixBase<float>&, kaldi::CuMatrix<float>*)

# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:403) xentropy_aux1_ vector =  [
#   -0.00652056 -0 -0 
#   -5.18894 -0 -0 
#   -0 -0.00203865 -0 
#   -0 -5.63375 -0 
#   -0 -0 0 
#   -0 -0 0 ]


# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:404) cross_entropy1 (CE1) =  10.8312

# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:405) xentropy_aux2_ vector =  [
#   -0.0775177 -0.00109753 -0.0708222 
#   -2.65171 -1.03168e-10 -0.0806583 
#   -0.000112724 -0.0527787 -7.39374e-05 
#   -1.73453 -0.0562561 -0.0411507 
#   -0.0123612 -5.26832 0 
#   -5.39663e-06 -24.9189 0 ]


# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:406) cross_entropy2 (CE2) =  34.9663

# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:407) T/S Loss (rho*CE1+(1-rho)*CE2) =  30.1393

# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:408) entropy = -0

# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:465) diff =  [
#   -0.115695 0.120349 -0.00465414 
#   -1.68009 1.6874 -0.00731674 
#   0.067127 -0.0825852 0.0154582 
#   -0.573368 -0.140892 0.71426 
#   -0.000775774 -0.294656 0.295431 
#   -4.17945e-07 -1.59948 1.59948 ]


# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:354) grad =  [
#   -0.115695 0.120349 -0.00465414 
#   -1.68009 1.6874 -0.00731674 
#   0.067127 -0.0825852 0.0154582 
#   -0.573368 -0.140892 0.71426 
#   -0.000775774 -0.294656 0.295431 
#   -4.17945e-07 -1.59948 1.59948 ]


# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:364) ### After 0 frames,
# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:365) ### Forward propagation buffer content :
# [0] output of <Input>  ( min -13.1945, max 12.3055, mean 0.780497, variance 67.1269, skewness -0.11369, kurtosis -1.33857 ) 
# [1] output of <AffineTransform> ( min -26.6987, max 38.9556, mean -0.189907, variance 328.55, skewness 0.646944, kurtosis -0.511242 ) 
# [2] output of <Softmax> ( min 1.41891e-25, max 1, mean 0.333333, variance 0.220257, skewness 0.707086, kurtosis -1.49991 ) 

# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:367) ### Backward propagation buffer content :
# [0] diff of <Input>  ( min -3.36219, max 4.3459, mean 0.366727, variance 2.3745, skewness 0.936338, kurtosis 2.05549 ) 
# [1] diff-output of <AffineTransform> ( min -1.68009, max 1.6874, mean -9.4179e-10, variance 0.658839, skewness 0.0244622, kurtosis 0.781983 ) 
# [2] diff-output of <Softmax> ( min -1.68009, max 1.6874, mean -9.4179e-10, variance 0.658839, skewness 0.0244622, kurtosis 0.781983 ) 

# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:368) ### Gradient stats :
# Component 1 : <AffineTransform>, 
#   linearity_grad ( min -42.4699, max 21.794, mean -5.08626e-07, variance 651.827, skewness -0.688349, kurtosis -1.3924 ) , lr-coef 1, max-norm 0
#   bias_grad ( min -2.3028, max 2.61266, mean 0, variance 4.07497, skewness 0.226635, kurtosis -1.5 ) , lr-coef 1
# Component 2 : <Softmax>, 

# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:389) ### After 6 frames,
# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:390) ### Forward propagation buffer content :
# [0] output of <Input>  ( min -13.1945, max 12.3055, mean 0.780497, variance 67.1269, skewness -0.11369, kurtosis -1.33857 ) 
# [1] output of <AffineTransform> ( min -26.6987, max 38.9556, mean -0.189907, variance 328.55, skewness 0.646944, kurtosis -0.511242 ) 
# [2] output of <Softmax> ( min 1.41891e-25, max 1, mean 0.333333, variance 0.220257, skewness 0.707086, kurtosis -1.49991 ) 

# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:392) ### Backward propagation buffer content :
# [0] diff of <Input>  ( min -3.36219, max 4.3459, mean 0.366727, variance 2.3745, skewness 0.936338, kurtosis 2.05549 ) 
# [1] diff-output of <AffineTransform> ( min -1.68009, max 1.6874, mean -9.4179e-10, variance 0.658839, skewness 0.0244622, kurtosis 0.781983 ) 
# [2] diff-output of <Softmax> ( min -1.68009, max 1.6874, mean -9.4179e-10, variance 0.658839, skewness 0.0244622, kurtosis 0.781983 ) 

# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:393) ### Gradient stats :
# Component 1 : <AffineTransform>, 
#   linearity_grad ( min -42.4699, max 21.794, mean -5.08626e-07, variance 651.827, skewness -0.688349, kurtosis -1.3924 ) , lr-coef 1, max-norm 0
#   bias_grad ( min -2.3028, max 2.61266, mean 0, variance 4.07497, skewness 0.226635, kurtosis -1.5 ) , lr-coef 1
# Component 2 : <Softmax>, 

# LOG (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:401) Done 1 files, 0 with no tgt_mats, 0 with other errors. [TRAINING, NOT-RANDOMIZED, 0.000147319 min, fps678.891]
# LOG (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:414) AvgLoss: 5.02321 (AvgTS - AvgTargetEnt), [AvgTS: 5.02321, AvgXent (Hard):  1.80521, AvgXent (Soft):  5.82771, rho:     0.2, AvgTargetEnt: 0]
# FRAME_ACCURACY >> 66.6667% <<


# ----------------- 2-layer dnn (affine xform + sigmoid + affine xform + softmax) -----------------
nnet_proto=exp/dnn/proto2L    # student proto
nnet_init=exp/dnn/nnet2L.in   # student nnet
nnet_out=exp/dnn/nnet2L.out   # output of student nnet
post1=exp/dnn/post1.txt
post2=exp/dnn/post2.txt
T=2.0
rho=0.2

# Create the student DNN
python $utils/nnet/make_nnet_proto.py 5 3 1 3 > $nnet_proto
nnet-initialize --binary=false $nnet_proto $nnet_init

echo "Features"
cat $feats

echo "Nnet"
cat $nnet_init

# Feedforward the features through the teacher network using a temperature based softmax to get soft labels
nnet-forward --softmax-temperature=$T $nnet_teacher ark:$feats ark,t:-| feat-to-post ark:- ark,t:$post2

# Train the student DNN with Teacher-Student loss
nnet-train-frmshuff --verbose=4 --randomize=false  --use-gpu=no --objective-function=ts --minibatch-size=6 --softmax-temperature=$T --rho=$rho \
  ark:$feats ark:$post1 ark:$post2 $nnet_init $nnet_out

# Expected Output (2 layer, T = 2, rho  = 0.2)
# nnet-initialize --binary=false exp/dnn/proto2L exp/dnn/nnet2L.in 
# VLOG[1] (nnet-initialize:Init():nnet-nnet.cc:378) <NnetProto>
# VLOG[1] (nnet-initialize:Init():nnet-nnet.cc:378) <AffineTransform> <InputDim> 5 <OutputDim> 3 <BiasMean> -2.000000 <BiasRange> 4.000000 <ParamStddev> 0.505181 <MaxNorm> 0.000000
# VLOG[1] (nnet-initialize:Init():nnet-nnet.cc:378) <Sigmoid> <InputDim> 3 <OutputDim> 3
# VLOG[1] (nnet-initialize:Init():nnet-nnet.cc:378) <AffineTransform> <InputDim> 3 <OutputDim> 3 <BiasMean> 0.000000 <BiasRange> 0.000000 <ParamStddev> 2.020726 <LearnRateCoef> 1.000000 <BiasLearnRateCoef> 0.100000
# VLOG[1] (nnet-initialize:Init():nnet-nnet.cc:378) <Softmax> <InputDim> 3 <OutputDim> 3
# VLOG[1] (nnet-initialize:Init():nnet-nnet.cc:378) </NnetProto>
# LOG (nnet-initialize:main():nnet-initialize.cc:64) Written initialized model to exp/dnn/nnet2L.in
# Features
# s1u1 [ 
#    -9.2396   -7.4065    -13.1945  -8.7807   -9.5492 
#   -11.8494  -10.6132   -9.5155   -4.9394   -6.0834 
#   -1.9090    4.2920     1.0259   -0.0892    1.0108 
#   -0.2899   -0.1756     2.1068    1.9927    2.0042 
#   10.9496    8.2924    11.0143   12.3055   10.6914 
#   11.4633   11.0280     9.5709   10.4156    8.8866 ]Nnet
# <Nnet> 
# <AffineTransform> 3 5 
# <LearnRateCoef> 1 <BiasLearnRateCoef> 1 <MaxNorm> 0  [
#   0.5971861 -0.1485795 -0.8406976 0.0259598 0.0149312 
#   0.02271999 -0.1298956 -0.1417916 -0.1553033 -0.09070534 
#   0.7647071 -0.4271501 -0.1589169 0.6290438 0.03965995 ]
#  [ -3.203336 -2.703211 -1.149163 ]
# <Sigmoid> 3 3 
# <AffineTransform> 3 3 
# <LearnRateCoef> 1 <BiasLearnRateCoef> 0.1 <MaxNorm> 0  [
#   1.209901 -0.5882571 4.255136 
#   0.375723 0.6421427 -1.406606 
#   0.4787499 0.6627153 1.329886 ]
#  [ 0 0 0 ]
# <Softmax> 3 3 
# </Nnet> 
# feat-to-post ark:- ark,t:exp/dnn/post2.txt 
# nnet-forward --softmax-temperature=2.0 exp/dnn/nnet2L.in ark:common/feats.txt ark,t:- 
# LOG (nnet-forward:SelectGpuId():cu-device.cc:83) Manually selected to compute on CPU.
# LOG (nnet-forward:main():nnet-forward.cc:221) Done 1 files in 1.98285e-06min, (fps 51254.2)
# LOG (feat-to-post:main():feat-to-post.cc:71) Converted 1 alignments.
# nnet-train-frmshuff --verbose=4 --randomize=false --use-gpu=no --objective-function=ts --minibatch-size=6 --softmax-temperature=2.0 --rho=0.2 ark:common/feats.txt ark:exp/dnn/post1.txt ark:exp/dnn/post2.txt exp/dnn/nnet2L.in exp/dnn/nnet2L.out 
# LOG (nnet-train-frmshuff:SelectGpuId():cu-device.cc:83) Manually selected to compute on CPU.
# LOG (nnet-train-frmshuff:Init():nnet-randomizer.cc:31) Seeding by srand with : 777
# LOG (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:190) Objective Function = ts

# LOG (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:193) TRAINING STARTED
# VLOG[3] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:205) Reading s1u1
# LOG (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:273) Finished filling randomizer. num done = 1, num_no_tgt_mat = 0, num_no_frame_wts = 0

# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:295) Mini-batch loop

# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:300) nnet_in =  [
#   -9.2396 -7.4065 -13.1945 -8.7807 -9.5492 
#   -11.8494 -10.6132 -9.5155 -4.9394 -6.0834 
#   -1.909 4.292 1.0259 -0.0892 1.0108 
#   -0.2899 -0.1756 2.1068 1.9927 2.0042 
#   10.9496 8.2924 11.0143 12.3055 10.6914 
#   11.4633 11.028 9.5709 10.4156 8.8866 ]


# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:301) nnet_out =  [
#   0.257981 0.349255 0.392764 
#   0.198896 0.391313 0.409791 
#   0.335251 0.327796 0.336953 
#   0.719666 0.0665316 0.213802 
#   0.945978 0.00328736 0.0507342 
#   0.946023 0.00328568 0.0506912 ]


# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:323) nnet_in =  [
#   -9.2396 -7.4065 -13.1945 -8.7807 -9.5492 
#   -11.8494 -10.6132 -9.5155 -4.9394 -6.0834 
#   -1.909 4.292 1.0259 -0.0892 1.0108 
#   -0.2899 -0.1756 2.1068 1.9927 2.0042 
#   10.9496 8.2924 11.0143 12.3055 10.6914 
#   11.4633 11.028 9.5709 10.4156 8.8866 ]


# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:324) Temperature of nnet1 =  
#   Temperature = 1

# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:325) Temperature of nnet2 =  
#   Temperature = 2

# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:458) nnet_out1 =  [
#   0.257981 0.349255 0.392764 
#   0.198896 0.391313 0.409791 
#   0.335251 0.327796 0.336953 
#   0.719666 0.0665316 0.213802 
#   0.945978 0.00328736 0.0507342 
#   0.946023 0.00328568 0.0506912 ]


# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:459) nnet_tgt =  [
#   1 0 0 
#   1 0 0 
#   0 1 0 
#   0 1 0 
#   0 0 1 
#   0 0 1 ]


# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:460) nnet_out2 =  [
#   0.294342 0.342476 0.363182 
#   0.26055 0.36546 0.37399 
#   0.334297 0.330559 0.335144 
#   0.540801 0.164432 0.294767 
#   0.774873 0.0456787 0.179449 
#   0.774945 0.0456701 0.179385 ]


# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:461) nnet_tgt2 =  [
#   0.294342 0.342476 0.363182 
#   0.26055 0.36546 0.37399 
#   0.334297 0.330559 0.335144 
#   0.540801 0.164432 0.294767 
#   0.774873 0.0456787 0.179449 
#   0.774945 0.0456702 0.179385 ]


# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:402) virtual void kaldi::nnet1::TS::Eval2(const kaldi::VectorBase<float>&, const kaldi::CuMatrixBase<float>&, const kaldi::CuMatrixBase<float>&, const kaldi::CuMatrixBase<float>&, const kaldi::CuMatrixBase<float>&, kaldi::CuMatrix<float>*)

# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:403) xentropy_aux1_ vector =  [
#   -1.35487 -0 -0 
#   -1.61497 -0 -0 
#   -0 -1.11536 -0 
#   -0 -2.71008 -0 
#   -0 -0 -2.98115 
#   -0 -0 -2.982 ]


# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:404) cross_entropy1 (CE1) =  12.7584

# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:405) xentropy_aux2_ vector =  [
#   -0.359984 -0.366981 -0.367849 
#   -0.35043 -0.367871 -0.367829 
#   -0.366298 -0.365919 -0.366378 
#   -0.332432 -0.296842 -0.360078 
#   -0.197636 -0.14097 -0.308269 
#   -0.197583 -0.140952 -0.308223 ]


# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:406) cross_entropy2 (CE2) =  5.56252

# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:407) T/S Loss (rho*CE1+(1-rho)*CE2) =  7.00171

# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:408) entropy = -0

# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:465) diff =  [
#   -0.148404 0.069851 0.0785528 
#   -0.160221 0.0782625 0.0819583 
#   0.0670502 -0.134441 0.0673905 
#   0.143933 -0.186694 0.0427605 
#   0.189196 0.000657473 -0.189853 
#   0.189205 0.00065713 -0.189862 ]


# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:354) grad =  [
#   -0.148404 0.069851 0.0785528 
#   -0.160221 0.0782625 0.0819583 
#   0.0670502 -0.134441 0.0673905 
#   0.143933 -0.186694 0.0427605 
#   0.189196 0.000657473 -0.189853 
#   0.189205 0.00065713 -0.189862 ]


# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:364) ### After 0 frames,
# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:365) ### Forward propagation buffer content :
# [0] output of <Input>  ( min -13.1945, max 12.3055, mean 0.780497, variance 67.1269, skewness -0.11369, kurtosis -1.33857 ) 
# [1] output of <AffineTransform> ( min -8.8564, max 10.0963, mean -2.40673, variance 28.9713, skewness 0.923685, kurtosis -0.0985193 ) 
# [2] output of <Sigmoid> ( min 0.000142447, max 0.999959, mean 0.299571, variance 0.162334, skewness 0.84819, kurtosis -1.07261 ) 
# [3] output of <AffineTransform> ( min -1.40585, max 4.25808, mean 0.777241, variance 2.21828, skewness 1.01702, kurtosis 0.893959 ) 
# [4] output of <Softmax> ( min 0.00328568, max 0.946023, mean 0.333333, variance 0.0777694, skewness 0.967908, kurtosis 0.180521 ) 

# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:367) ### Backward propagation buffer content :
# [0] diff of <Input>  ( min -0.0967985, max 0.174632, mean 0.0061353, variance 0.00205055, skewness 2.23668, kurtosis 7.1859 ) 
# [1] diff-output of <AffineTransform> ( min -0.0254888, max 0.227896, mean 0.0139893, variance 0.00282603, skewness 3.58487, kurtosis 11.5877 ) 
# [2] diff-output of <Sigmoid> ( min -0.682851, max 0.931927, mean 0.0648024, variance 0.160379, skewness 0.156682, kurtosis -0.186992 ) 
# [3] diff-output of <AffineTransform> ( min -0.189862, max 0.189205, mean 1.30062e-08, variance 0.0166546, skewness -0.282702, kurtosis -1.30459 ) 
# [4] diff-output of <Softmax> ( min -0.189862, max 0.189205, mean 1.30062e-08, variance 0.0166546, skewness -0.282702, kurtosis -1.30459 ) 

# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:368) ### Gradient stats :
# Component 1 : <AffineTransform>, 
#   linearity_grad ( min -0.602578, max 0.492008, mean 0.0134498, variance 0.158117, skewness -0.384141, kurtosis -1.41536 ) , lr-coef 1, max-norm 0
#   bias_grad ( min -0.0286143, max 0.233038, mean 0.0839358, variance 0.0120784, skewness 0.462095, kurtosis -1.5 ) , lr-coef 1
# Component 2 : <Sigmoid>, 
# Component 3 : <AffineTransform>, 
#   linearity_grad ( min -0.360725, max 0.440224, mean 2.73188e-08, variance 0.0525514, skewness 0.203216, kurtosis -0.585672 ) , lr-coef 1, max-norm 0
#   bias_grad ( min -0.171706, max 0.280759, mean 6.95387e-08, variance 0.0400672, skewness 0.655504, kurtosis -1.5 ) , lr-coef 0.1
# Component 4 : <Softmax>, 

# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:389) ### After 6 frames,
# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:390) ### Forward propagation buffer content :
# [0] output of <Input>  ( min -13.1945, max 12.3055, mean 0.780497, variance 67.1269, skewness -0.11369, kurtosis -1.33857 ) 
# [1] output of <AffineTransform> ( min -8.8564, max 10.0963, mean -2.40673, variance 28.9713, skewness 0.923685, kurtosis -0.0985193 ) 
# [2] output of <Sigmoid> ( min 0.000142447, max 0.999959, mean 0.299571, variance 0.162334, skewness 0.84819, kurtosis -1.07261 ) 
# [3] output of <AffineTransform> ( min -1.40585, max 4.25808, mean 0.777241, variance 2.21828, skewness 1.01702, kurtosis 0.893959 ) 
# [4] output of <Softmax> ( min 0.00328568, max 0.946023, mean 0.333333, variance 0.0777694, skewness 0.967908, kurtosis 0.180521 ) 

# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:392) ### Backward propagation buffer content :
# [0] diff of <Input>  ( min -0.0967985, max 0.174632, mean 0.0061353, variance 0.00205055, skewness 2.23668, kurtosis 7.1859 ) 
# [1] diff-output of <AffineTransform> ( min -0.0254888, max 0.227896, mean 0.0139893, variance 0.00282603, skewness 3.58487, kurtosis 11.5877 ) 
# [2] diff-output of <Sigmoid> ( min -0.682851, max 0.931927, mean 0.0648024, variance 0.160379, skewness 0.156682, kurtosis -0.186992 ) 
# [3] diff-output of <AffineTransform> ( min -0.189862, max 0.189205, mean 1.30062e-08, variance 0.0166546, skewness -0.282702, kurtosis -1.30459 ) 
# [4] diff-output of <Softmax> ( min -0.189862, max 0.189205, mean 1.30062e-08, variance 0.0166546, skewness -0.282702, kurtosis -1.30459 ) 

# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:393) ### Gradient stats :
# Component 1 : <AffineTransform>, 
#   linearity_grad ( min -0.602578, max 0.492008, mean 0.0134498, variance 0.158117, skewness -0.384141, kurtosis -1.41536 ) , lr-coef 1, max-norm 0
#   bias_grad ( min -0.0286143, max 0.233038, mean 0.0839358, variance 0.0120784, skewness 0.462095, kurtosis -1.5 ) , lr-coef 1
# Component 2 : <Sigmoid>, 
# Component 3 : <AffineTransform>, 
#   linearity_grad ( min -0.360725, max 0.440224, mean 2.73188e-08, variance 0.0525514, skewness 0.203216, kurtosis -0.585672 ) , lr-coef 1, max-norm 0
#   bias_grad ( min -0.171706, max 0.280759, mean 6.95387e-08, variance 0.0400672, skewness 0.655504, kurtosis -1.5 ) , lr-coef 0.1
# Component 4 : <Softmax>, 

# LOG (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:401) Done 1 files, 0 with no tgt_mats, 0 with other errors. [TRAINING, NOT-RANDOMIZED, 0.000217382 min, fps460.053]
# LOG (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:414) AvgLoss: 1.16695 (AvgTS - AvgTargetEnt), [AvgTS: 1.16695, AvgXent (Hard):  2.12641, AvgXent (Soft):  0.927087, rho:     0.2, AvgTargetEnt: 0]
# FRAME_ACCURACY >> 0% <<
