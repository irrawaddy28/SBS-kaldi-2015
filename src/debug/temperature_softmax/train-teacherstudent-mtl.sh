#!/bin/bash

. ./path.sh
utils=../../../egs/wsj/s5/utils

nnet_teacher=common/nnet.teacher
feats=common/feats.txt

# ----------------- 2-layer MTL (affine xform + sigmoid + affine xform + softmax) -----------------
nnet_proto=exp/mtl/protoMTL    # student proto
nnet_init=exp/mtl/nnetMTL.in   # student nnet
nnet_out=exp/mtl/nnetMTL.out   # output of student nnet
post1=exp/mtl/post1.txt
post2=exp/mtl/post2.txt
T=2    #orig 2.0
rho=0.2  #orig 0.2

# student proto design
block_softmax_dims="3:3"
feat_dim=5
num_leaves=6 # sum of block softmax dims
num_hid_layers=1
num_hid_neurons=3

# Create the student MTL network
python $utils/nnet/make_nnet_proto.py --block-softmax-dims=${block_softmax_dims} $feat_dim $num_leaves $num_hid_layers $num_hid_neurons > $nnet_proto
nnet-initialize --binary=false $nnet_proto $nnet_init

# Feedforward the features through the teacher network using a temperature based softmax to get soft labels
# nnet-forward --softmax-temperature=$T $nnet_teacher ark:$feats ark,t:-| feat-to-post ark:- ark,t:$post2
post2="ark:nnet-forward --softmax-temperature=$T $nnet_teacher ark:$feats ark:- | feat-to-post ark:- ark:- |"


# Train the student MTL network with Teacher-Student loss in the first softmax and Cross Entropy loss in the second softmax
nnet-train-frmshuff --verbose=4 --use-gpu=yes --minibatch-size=6 --randomize=false\
  --objective-function='multitask,ts,3,1.0,xent,3,1.0' --softmax-temperature=$T --rho=$rho \
  ark:$feats ark:$post1 "$post2" $nnet_init $nnet_out


 
# Expected Output (T = 2, rho = 0.2)
# > ./train-teacherstudent-mtl.sh 

# nnet-initialize --binary=false exp/mtl/protoMTL exp/mtl/nnetMTL.in 
# VLOG[1] (nnet-initialize:Init():nnet-nnet.cc:378) <NnetProto>
# VLOG[1] (nnet-initialize:Init():nnet-nnet.cc:378) <AffineTransform> <InputDim> 5 <OutputDim> 3 <BiasMean> -2.000000 <BiasRange> 4.000000 <ParamStddev> 0.505181 <MaxNorm> 0.000000
# VLOG[1] (nnet-initialize:Init():nnet-nnet.cc:378) <Sigmoid> <InputDim> 3 <OutputDim> 3
# VLOG[1] (nnet-initialize:Init():nnet-nnet.cc:378) <AffineTransform> <InputDim> 3 <OutputDim> 6 <BiasMean> 0.000000 <BiasRange> 0.000000 <ParamStddev> 1.649916 <LearnRateCoef> 1.000000 <BiasLearnRateCoef> 0.100000
# VLOG[1] (nnet-initialize:Init():nnet-nnet.cc:378) <BlockSoftmax> <InputDim> 6 <OutputDim> 6 <BlockDims> 3:3
# VLOG[1] (nnet-initialize:Init():nnet-nnet.cc:378) </NnetProto>
# LOG (nnet-initialize:main():nnet-initialize.cc:64) Written initialized model to exp/mtl/nnetMTL.in
# nnet-train-frmshuff --verbose=4 --use-gpu=yes --minibatch-size=6 --randomize=false --objective-function=multitask,ts,3,1.0,xent,3,1.0 --softmax-temperature=2 --rho=0.2 ark:common/feats.txt ark:exp/mtl/post1.txt 'ark:nnet-forward --softmax-temperature=2 common/nnet.teacher ark:common/feats.txt ark:- | feat-to-post ark:- ark:- |' exp/mtl/nnetMTL.in exp/mtl/nnetMTL.out 
# WARNING (nnet-train-frmshuff:SelectGpuId():cu-device.cc:158) Suggestion: use 'nvidia-smi -c 1' to set compute exclusive mode
# LOG (nnet-train-frmshuff:SelectGpuIdAuto():cu-device.cc:280) Selecting from 2 GPUs
# LOG (nnet-train-frmshuff:SelectGpuIdAuto():cu-device.cc:295) cudaSetDevice(0): GeForce GTX 780	free:1623M, used:1394M, total:3017M, free/total:0.538027
# LOG (nnet-train-frmshuff:SelectGpuIdAuto():cu-device.cc:295) cudaSetDevice(1): GeForce GTX 780	free:2383M, used:637M, total:3020M, free/total:0.789042
# LOG (nnet-train-frmshuff:SelectGpuIdAuto():cu-device.cc:344) Trying to select device: 1 (automatically), mem_ratio: 0.789042
# LOG (nnet-train-frmshuff:SelectGpuIdAuto():cu-device.cc:363) Success selecting device 1 free mem ratio: 0.789042
# LOG (nnet-train-frmshuff:FinalizeActiveGpu():cu-device.cc:202) The active GPU is [1]: GeForce GTX 780	free:2365M, used:655M, total:3020M, free/total:0.782918 version 3.5
# LOG (nnet-train-frmshuff:PrintMemoryUsage():cu-device.cc:379) Memory used: 0 bytes.
# LOG (nnet-train-frmshuff:Init():nnet-randomizer.cc:31) Seeding by srand with : 777
# LOG (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:190) Objective Function = multitask,ts,3,1.0,xent,3,1.0

# LOG (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:193) TRAINING STARTED
# VLOG[3] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:205) Reading s1u1
# feat-to-post ark:- ark:- 
# nnet-forward --softmax-temperature=2 common/nnet.teacher ark:common/feats.txt ark:- 
# LOG (nnet-forward:SelectGpuId():cu-device.cc:83) Manually selected to compute on CPU.
# LOG (nnet-forward:main():nnet-forward.cc:221) Done 1 files in 1.56959e-06min, (fps 64362.7)
# LOG (feat-to-post:main():feat-to-post.cc:71) Converted 1 alignments.
# LOG (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:273) Finished filling randomizer. num done = 1, num_no_tgt_mat = 0, num_no_frame_wts = 0

# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:295) Mini-batch loop

# VLOG[3] (nnet-train-frmshuff:PropagateFnc():nnet/nnet-activation.h:139) Feedfwd through blocksoftmax 

# VLOG[3] (nnet-train-frmshuff:PropagateFnc():nnet/nnet-activation.h:145)  T = 1, Block = 0, num_frames =  6, num_pdf =  3

# VLOG[3] (nnet-train-frmshuff:PropagateFnc():nnet/nnet-activation.h:192) Ready to exit

# VLOG[3] (nnet-train-frmshuff:PropagateFnc():nnet/nnet-activation.h:145)  T = 1, Block = 1, num_frames =  6, num_pdf =  3

# VLOG[3] (nnet-train-frmshuff:PropagateFnc():nnet/nnet-activation.h:192) Ready to exit

# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:300) nnet_in =  [
#   -9.2396 -7.4065 -13.1945 -8.7807 -9.5492 
#   -11.8494 -10.6132 -9.5155 -4.9394 -6.0834 
#   -1.909 4.292 1.0259 -0.0892 1.0108 
#   -0.2899 -0.1756 2.1068 1.9927 2.0042 
#   10.9496 8.2924 11.0143 12.3055 10.6914 
#   11.4633 11.028 9.5709 10.4156 8.8866 ]


# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:301) nnet_out =  [
#   0.271 0.347043 0.381958 0.45458 0.45765 0.0877699 
#   0.220165 0.382573 0.397262 0.237583 0.690322 0.0720953 
#   0.334902 0.328809 0.336289 0.329809 0.350748 0.319443 
#   0.660368 0.0945027 0.245129 0.262213 0.332462 0.405325 
#   0.907799 0.00891654 0.0832847 0.177553 0.272997 0.549449 
#   0.907858 0.00891305 0.0832292 0.177892 0.272279 0.54983 ]


# VLOG[3] (nnet-train-frmshuff:PropagateFnc():nnet/nnet-activation.h:139) Feedfwd through blocksoftmax 

# VLOG[3] (nnet-train-frmshuff:PropagateFnc():nnet/nnet-activation.h:145)  T = 2, Block = 0, num_frames =  6, num_pdf =  3

# VLOG[3] (nnet-train-frmshuff:PropagateFnc():nnet/nnet-activation.h:171) in_bl (after scale) =  [
#   0.515666 0.762995 0.858857 
#   -0.0750583 0.477485 0.515159 
#   0.023034 0.00467302 0.0271688 
#   1.47246 -0.471714 0.481444 
#   3.47524 -1.14788 1.08648 
#   3.47671 -1.14687 1.08722 ]


# VLOG[3] (nnet-train-frmshuff:PropagateFnc():nnet/nnet-activation.h:172) inT   (after scale) =  [
#   0.257833 0.381497 0.429429 
#   -0.0375292 0.238742 0.25758 
#   0.011517 0.00233651 0.0135844 
#   0.736228 -0.235857 0.240722 
#   1.73762 -0.573938 0.543241 
#   1.73835 -0.573433 0.543608 ]


# VLOG[3] (nnet-train-frmshuff:PropagateFnc():nnet/nnet-activation.h:192) Ready to exit

# VLOG[3] (nnet-train-frmshuff:PropagateFnc():nnet/nnet-activation.h:145)  T = 2, Block = 1, num_frames =  6, num_pdf =  3

# VLOG[3] (nnet-train-frmshuff:PropagateFnc():nnet/nnet-activation.h:171) in_bl (after scale) =  [
#   -0.998137 -0.991405 -2.64279 
#   -0.307064 0.759579 -1.49959 
#   -0.0169244 0.0446305 -0.0488591 
#   -0.591247 -0.353877 -0.155713 
#   -1.37186 -0.941665 -0.242211 
#   -1.37391 -0.94826 -0.245477 ]


# VLOG[3] (nnet-train-frmshuff:PropagateFnc():nnet/nnet-activation.h:172) inT   (after scale) =  [
#   -0.998137 -0.991405 -2.64279 
#   -0.307064 0.759579 -1.49959 
#   -0.0169244 0.0446305 -0.0488591 
#   -0.591247 -0.353877 -0.155713 
#   -1.37186 -0.941665 -0.242211 
#   -1.37391 -0.94826 -0.245477 ]


# VLOG[3] (nnet-train-frmshuff:PropagateFnc():nnet/nnet-activation.h:192) Ready to exit

# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:345) nnet_in =  [
#   -9.2396 -7.4065 -13.1945 -8.7807 -9.5492 
#   -11.8494 -10.6132 -9.5155 -4.9394 -6.0834 
#   -1.909 4.292 1.0259 -0.0892 1.0108 
#   -0.2899 -0.1756 2.1068 1.9927 2.0042 
#   10.9496 8.2924 11.0143 12.3055 10.6914 
#   11.4633 11.028 9.5709 10.4156 8.8866 ]


# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:346) Temperature of nnet1 =  
#   softmax-dims 3 3 
#   softmax-temperature (of 1st block)  1

# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:347) Temperature of nnet2 =  
#   softmax-dims 3 3 
#   softmax-temperature (of 1st block)  2

# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:664) nnet_out1 =  [
#   0.271 0.347043 0.381958 0.45458 0.45765 0.0877699 
#   0.220165 0.382573 0.397262 0.237583 0.690322 0.0720953 
#   0.334902 0.328809 0.336289 0.329809 0.350748 0.319443 
#   0.660368 0.0945027 0.245129 0.262213 0.332462 0.405325 
#   0.907799 0.00891654 0.0832847 0.177553 0.272997 0.549449 
#   0.907858 0.00891305 0.0832292 0.177892 0.272279 0.54983 ]


# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:665) nnet_tgt1 =  [
#   1 0 0 0 0 0 
#   0 0 0 1 0 0 
#   0 1 0 0 0 0 
#   0 0 0 0 1 0 
#   0 0 1 0 0 0 
#   0 0 0 0 0 1 ]


# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:666) nnet_out2 =  [
#   0.301311 0.340974 0.357715 0.45458 0.45765 0.0877699 
#   0.273114 0.36002 0.366866 0.237583 0.690322 0.0720953 
#   0.334121 0.331067 0.334812 0.329809 0.350748 0.319443 
#   0.50313 0.190331 0.306539 0.262213 0.332462 0.405325 
#   0.713267 0.0706896 0.216043 0.177553 0.272997 0.549449 
#   0.713335 0.0706802 0.215985 0.177892 0.272279 0.54983 ]


# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:667) nnet_tgt2 =  [
#   0.996668 0.000423341 0.00290884 0 0 0 
#   0.995427 1.42847e-09 0.00457299 0 0 0 
#   3.54915e-05 0.999949 1.59372e-05 0 0 0 
#   0.367255 0.0195188 0.613226 0 0 0 
#   0.000484859 0.18416 0.815355 0 0 0 
#   2.62384e-07 0.999678 0.000321986 0 0 0 ]


# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:402) virtual void kaldi::nnet1::TS::Eval2(const kaldi::VectorBase<float>&, const kaldi::CuMatrixBase<float>&, const kaldi::CuMatrixBase<float>&, const kaldi::CuMatrixBase<float>&, const kaldi::CuMatrixBase<float>&, kaldi::CuMatrix<float>*)

# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:403) xentropy_aux1_ vector =  [
#   -1.30564 -0 -0 
#   -0 -0 -0 
#   -0 -1.11228 -0 
#   -0 -0 -0 
#   -0 -0 -2.48549 
#   -0 -0 -0 ]


# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:404) cross_entropy1 (CE1) =  4.90341

# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:405) xentropy_aux2_ vector =  [
#   -1.19562 -0.000455493 -0.00299033 
#   -0 -0 -0 
#   -3.89077e-05 -1.10538 -1.74382e-05 
#   -0 -0 -0 
#   -0.000163833 -0.487923 -1.24935 
#   -0 -0 -0 ]


# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:406) cross_entropy2 (CE2) =  4.04193

# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:407) T/S Loss (rho*CE1+(1-rho)*CE2) =  4.21423

# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:408) entropy = -0

# VLOG[4] (nnet-train-frmshuff:Eval():nnet-loss.cc:168) virtual void kaldi::nnet1::Xent::Eval(const kaldi::VectorBase<float>&, const kaldi::CuMatrixBase<float>&, const kaldi::CuMatrixBase<float>&, kaldi::CuMatrix<float>*)

# VLOG[4] (nnet-train-frmshuff:Eval():nnet-loss.cc:169) xentropy_aux_ vector =  [
#   -0 -0 -0 
#   -1.43724 -0 -0 
#   -0 -0 -0 
#   -0 -1.10123 -0 
#   -0 -0 -0 
#   -0 -0 -0.598147 ]


# VLOG[4] (nnet-train-frmshuff:Eval():nnet-loss.cc:170) cross_entropy =  3.13662

# VLOG[4] (nnet-train-frmshuff:Eval():nnet-loss.cc:171) entropy = -0

# VLOG[4] (nnet-train-frmshuff:Eval2():nnet-loss.cc:720) diff =  [
#   -1.25837 0.61429 0.644082 0 0 0 
#   -0 0 0 -0.762417 0.690322 0.0720953 
#   0.601517 -1.20445 0.602932 0 0 0 
#   0 0 -0 0.262213 -0.667538 0.405325 
#   1.32201 -0.179769 -1.14224 0 0 0 
#   0 -0 0 0.177892 0.272279 -0.45017 ]


# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:354) grad =  [
#   -1.25837 0.61429 0.644082 0 0 0 
#   -0 0 0 -0.762417 0.690322 0.0720953 
#   0.601517 -1.20445 0.602932 0 0 0 
#   0 0 -0 0.262213 -0.667538 0.405325 
#   1.32201 -0.179769 -1.14224 0 0 0 
#   0 -0 0 0.177892 0.272279 -0.45017 ]


# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:364) ### After 0 frames,
# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:365) ### Forward propagation buffer content :
# [0] output of <Input>  ( min -13.1945, max 12.3055, mean 0.780497, variance 67.1269, skewness -0.11369, kurtosis -1.33857 ) 
# [1] output of <AffineTransform> ( min -8.8564, max 10.0963, mean -2.40672, variance 28.9713, skewness 0.923685, kurtosis -0.0985193 ) 
# [2] output of <Sigmoid> ( min 0.000142447, max 0.999959, mean 0.299571, variance 0.162334, skewness 0.84819, kurtosis -1.07261 ) 
# [3] output of <AffineTransform> ( min -2.64279, max 3.47671, mean -0.0139365, variance 1.44231, skewness 0.959558, kurtosis 1.97216 ) 
# [4] output of <BlockSoftmax> ( min 0.00891305, max 0.907858, mean 0.333333, variance 0.045796, skewness 0.927075, kurtosis 0.887316 ) 

# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:367) ### Backward propagation buffer content :
# [0] diff of <Input>  ( min -0.170608, max 0.210615, mean 0.00086971, variance 0.00275232, skewness 0.954774, kurtosis 9.37329 ) 
# [1] diff-output of <AffineTransform> ( min -0.296259, max 0.270935, mean 0.00410318, variance 0.010193, skewness -0.367488, kurtosis 4.16883 ) 
# [2] diff-output of <Sigmoid> ( min -4.3781, max 4.12784, mean 0.180962, variance 3.65033, skewness -0.0798204, kurtosis 0.551537 ) 
# [3] diff-output of <AffineTransform> ( min -1.25837, max 1.32201, mean 4.13921e-09, variance 0.269072, skewness -0.474024, kurtosis 1.13991 ) 
# [4] diff-output of <BlockSoftmax> ( min -1.25837, max 1.32201, mean 4.13921e-09, variance 0.269072, skewness -0.474024, kurtosis 1.13991 ) 

# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:368) ### Gradient stats :
# Component 1 : <AffineTransform>, 
#   linearity_grad ( min -4.29419, max 3.80571, mean -0.200261, variance 7.31747, skewness -0.143229, kurtosis -1.23261 ) , lr-coef 1, max-norm 0
#   bias_grad ( min -0.320475, max 0.313334, mean 0.0246191, variance 0.0685415, skewness -0.313035, kurtosis -1.5 ) , lr-coef 1
# Component 2 : <Sigmoid>, 
# Component 3 : <AffineTransform>, 
#   linearity_grad ( min -1.20078, max 1.32764, mean 1.00376e-08, variance 0.439465, skewness -0.286965, kurtosis -0.411147 ) , lr-coef 1, max-norm 0
#   bias_grad ( min -0.769927, max 0.665157, mean 2.23517e-08, variance 0.206315, skewness -0.300105, kurtosis -0.78524 ) , lr-coef 0.1
# Component 4 : <BlockSoftmax>, 

# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:389) ### After 6 frames,
# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:390) ### Forward propagation buffer content :
# [0] output of <Input>  ( min -13.1945, max 12.3055, mean 0.780497, variance 67.1269, skewness -0.11369, kurtosis -1.33857 ) 
# [1] output of <AffineTransform> ( min -8.8564, max 10.0963, mean -2.40672, variance 28.9713, skewness 0.923685, kurtosis -0.0985193 ) 
# [2] output of <Sigmoid> ( min 0.000142447, max 0.999959, mean 0.299571, variance 0.162334, skewness 0.84819, kurtosis -1.07261 ) 
# [3] output of <AffineTransform> ( min -2.64279, max 3.47671, mean -0.0139365, variance 1.44231, skewness 0.959558, kurtosis 1.97216 ) 
# [4] output of <BlockSoftmax> ( min 0.00891305, max 0.907858, mean 0.333333, variance 0.045796, skewness 0.927075, kurtosis 0.887316 ) 

# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:392) ### Backward propagation buffer content :
# [0] diff of <Input>  ( min -0.170608, max 0.210615, mean 0.00086971, variance 0.00275232, skewness 0.954774, kurtosis 9.37329 ) 
# [1] diff-output of <AffineTransform> ( min -0.296259, max 0.270935, mean 0.00410318, variance 0.010193, skewness -0.367488, kurtosis 4.16883 ) 
# [2] diff-output of <Sigmoid> ( min -4.3781, max 4.12784, mean 0.180962, variance 3.65033, skewness -0.0798204, kurtosis 0.551537 ) 
# [3] diff-output of <AffineTransform> ( min -1.25837, max 1.32201, mean 4.13921e-09, variance 0.269072, skewness -0.474024, kurtosis 1.13991 ) 
# [4] diff-output of <BlockSoftmax> ( min -1.25837, max 1.32201, mean 4.13921e-09, variance 0.269072, skewness -0.474024, kurtosis 1.13991 ) 

# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:393) ### Gradient stats :
# Component 1 : <AffineTransform>, 
#   linearity_grad ( min -4.29419, max 3.80571, mean -0.200261, variance 7.31747, skewness -0.143229, kurtosis -1.23261 ) , lr-coef 1, max-norm 0
#   bias_grad ( min -0.320475, max 0.313334, mean 0.0246191, variance 0.0685415, skewness -0.313035, kurtosis -1.5 ) , lr-coef 1
# Component 2 : <Sigmoid>, 
# Component 3 : <AffineTransform>, 
#   linearity_grad ( min -1.20078, max 1.32764, mean 1.00376e-08, variance 0.439465, skewness -0.286965, kurtosis -0.411147 ) , lr-coef 1, max-norm 0
#   bias_grad ( min -0.769927, max 0.665157, mean 2.23517e-08, variance 0.206315, skewness -0.300105, kurtosis -0.78524 ) , lr-coef 0.1
# Component 4 : <BlockSoftmax>, 

# LOG (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:401) Done 1 files, 0 with no tgt_mats, 0 with other errors. [TRAINING, NOT-RANDOMIZED, 0.000667651 min, fps149.779]
# LOG (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:416) MultiTaskLoss, with 2 parallel loss functions.
# Loss 1, AvgLoss: 1.40474 (AvgTS - AvgTargetEnt), [AvgTS: 1.40474, AvgXent (Hard):  1.63447, AvgXent (Soft):  1.34731, rho:     0.2, AvgTargetEnt: 0]
# FRAME_ACCURACY >> 0% <<

# Loss 2, AvgLoss: 1.04554 (Xent - AvgTgtEnt), [AvgXent: 1.04554, Target interpolation mode: none, rho:  1, AvgTargetEnt: 0]
# FRAME_ACCURACY >> 33.3333% <<

# Loss (OVERALL), AvgLoss: 2.45028 (MultiTaskLoss), weights 1 1 , values 1.40474 1.04554 

# LOG (nnet-train-frmshuff:PrintProfile():cu-device.cc:405) -----
# [cudevice profile]
# CuMatrixBase::CopyFromMat(from CPU)	0.000189066s
# Destroy	0.00022006s
# CopyToVec	0.000222445s
# AddMatVec	0.000303507s
# Sum	0.000315189s
# AddMatMat	0.000332832s
# CuVector::SetZero	0.000383615s
# Set	0.000400305s
# FindRowMaxId	0.000505209s
# CuMatrix::SetZero	0.000525475s
# CuMatrixBase::CopyFromMat(from other CuMatrixBase)	0.000604391s
# CheckGpuHealth	0.00103617s
# CuVector::Resize	0.00107884s
# CuMatrix::CopyToMatD2H	0.001091s
# CuMatrix::Resize	0.00162101s
# Total GPU time:	0.0103276s (may involve some double-counting)
# -----
# LOG (nnet-train-frmshuff:PrintMemoryUsage():cu-device.cc:379) Memory used: 17825792 bytes.
