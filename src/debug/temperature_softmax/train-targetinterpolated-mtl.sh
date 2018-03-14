#!/bin/bash

. ./path.sh
utils=../../../egs/wsj/s5/utils

feats=common/feats.txt

# ----------------- 2-layer MTL (affine xform + sigmoid + affine xform + softmax) -----------------
nnet_proto=exp/mtl/protoMTL    # proto
nnet_init=exp/mtl/nnetMTL.in   # nnet
nnet_out=exp/mtl/nnetMTL.out   # output of nnet
post1=exp/mtl/post1.txt
rho=0.2  #0.2

# student proto design
block_softmax_dims="3:3"
feat_dim=5
num_leaves=6 # sum of block softmax dims
num_hid_layers=1
num_hid_neurons=3

# Create the student MTL network
python $utils/nnet/make_nnet_proto.py --block-softmax-dims=${block_softmax_dims} $feat_dim $num_leaves $num_hid_layers $num_hid_neurons > $nnet_proto
nnet-initialize --binary=false $nnet_proto $nnet_init


# Train the MTL network with CE loss in the first and second softmax
# For the first softmax, let CE = sum_k (d_k log(y_k) ), where d_k are ground truth targets
# The CE loss in the first softmax is computed w.r.t three different ground truth targets as follows:

# 1) Ground truth labels from corpus (this is simply the standard CE training)
#    Thus, if t_k are the labels from corpus, then d_k = t_k.
nnet-train-frmshuff --verbose=4 --use-gpu=yes --minibatch-size=6 --randomize=false\
 --objective-function='multitask,xent,3,1.0,xent,3,1.0' --tgt-interp-mode="none" --rho=$rho \
 ark:$feats ark:$post1 $nnet_init $nnet_out



# 2) Ground truth labels computed as interpolation between corpus labels and posterior of nnet
#    Thus, d_k = rho*t_k + (1-rho)*y_k
nnet-train-frmshuff --verbose=4 --use-gpu=yes --minibatch-size=6 --randomize=false\
 --objective-function='multitask,xent,3,1.0,xent,3,1.0' --tgt-interp-mode="hard" --rho=$rho \
 ark:$feats ark:$post1 $nnet_init $nnet_out


# 3) Ground truth labels computed as interpolation between corpus labels and 1-hot posterior of nnet
#    Thus, d_k = rho*t_k + (1-rho)*1[k = argmax y], where y = [y_1 y_2 ... y_K]
nnet-train-frmshuff --verbose=4 --use-gpu=yes --minibatch-size=6 --randomize=false\
  --objective-function='multitask,xent,3,1.0,xent,3,1.0' --tgt-interp-mode="soft" --rho=$rho \
  ark:$feats ark:$post1 $nnet_init $nnet_out




# Expected output
# > ./train-targetinterpolated-mtl.sh 

# nnet-initialize --binary=false exp/mtl/protoMTL exp/mtl/nnetMTL.in 
# VLOG[1] (nnet-initialize:Init():nnet-nnet.cc:378) <NnetProto>
# VLOG[1] (nnet-initialize:Init():nnet-nnet.cc:378) <AffineTransform> <InputDim> 5 <OutputDim> 3 <BiasMean> -2.000000 <BiasRange> 4.000000 <ParamStddev> 0.505181 <MaxNorm> 0.000000
# VLOG[1] (nnet-initialize:Init():nnet-nnet.cc:378) <Sigmoid> <InputDim> 3 <OutputDim> 3
# VLOG[1] (nnet-initialize:Init():nnet-nnet.cc:378) <AffineTransform> <InputDim> 3 <OutputDim> 6 <BiasMean> 0.000000 <BiasRange> 0.000000 <ParamStddev> 1.649916 <LearnRateCoef> 1.000000 <BiasLearnRateCoef> 0.100000
# VLOG[1] (nnet-initialize:Init():nnet-nnet.cc:378) <BlockSoftmax> <InputDim> 6 <OutputDim> 6 <BlockDims> 3:3
# VLOG[1] (nnet-initialize:Init():nnet-nnet.cc:378) </NnetProto>
# LOG (nnet-initialize:main():nnet-initialize.cc:64) Written initialized model to exp/mtl/nnetMTL.in

# Target Interp = none
# nnet-train-frmshuff --verbose=4 --use-gpu=yes --minibatch-size=6 --randomize=false --objective-function=multitask,xent,3,1.0,xent,3,1.0 --tgt-interp-mode=none --rho=0.2 ark:common/feats.txt ark:exp/mtl/post1.txt exp/mtl/nnetMTL.in exp/mtl/nnetMTL.out 
# LOG (nnet-train-frmshuff:SelectGpuIdAuto():cu-device.cc:280) Selecting from 1 GPUs
# LOG (nnet-train-frmshuff:SelectGpuIdAuto():cu-device.cc:295) cudaSetDevice(0): Tesla K40c	free:11389M, used:82M, total:11471M, free/total:0.992841
# LOG (nnet-train-frmshuff:SelectGpuIdAuto():cu-device.cc:344) Trying to select device: 0 (automatically), mem_ratio: 0.992841
# LOG (nnet-train-frmshuff:SelectGpuIdAuto():cu-device.cc:363) Success selecting device 0 free mem ratio: 0.992841
# LOG (nnet-train-frmshuff:FinalizeActiveGpu():cu-device.cc:202) The active GPU is [0]: Tesla K40c	free:11372M, used:99M, total:11471M, free/total:0.991359 version 3.5
# LOG (nnet-train-frmshuff:PrintMemoryUsage():cu-device.cc:379) Memory used: 0 bytes.
# LOG (nnet-train-frmshuff:Init():nnet-randomizer.cc:31) Seeding by srand with : 777
# LOG (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:196) Objective Function = multitask,xent,3,1.0,xent,3,1.0

# LOG (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:199) TRAINING STARTED
# VLOG[3] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:211) Reading s1u1
# LOG (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:279) Finished filling randomizer. num done = 1, num_no_tgt_mat = 0, num_no_frame_wts = 0

# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:301) Mini-batch loop

# VLOG[3] (nnet-train-frmshuff:PropagateFnc():nnet/nnet-activation.h:139) Feedfwd through blocksoftmax 

# VLOG[3] (nnet-train-frmshuff:PropagateFnc():nnet/nnet-activation.h:145)  T = 1, Block = 0, num_frames =  6, num_pdf =  3

# VLOG[3] (nnet-train-frmshuff:PropagateFnc():nnet/nnet-activation.h:192) Ready to exit

# VLOG[3] (nnet-train-frmshuff:PropagateFnc():nnet/nnet-activation.h:145)  T = 1, Block = 1, num_frames =  6, num_pdf =  3

# VLOG[3] (nnet-train-frmshuff:PropagateFnc():nnet/nnet-activation.h:192) Ready to exit

# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:306) nnet_in =  [
#   -9.2396 -7.4065 -13.1945 -8.7807 -9.5492 
#   -11.8494 -10.6132 -9.5155 -4.9394 -6.0834 
#   -1.909 4.292 1.0259 -0.0892 1.0108 
#   -0.2899 -0.1756 2.1068 1.9927 2.0042 
#   10.9496 8.2924 11.0143 12.3055 10.6914 
#   11.4633 11.028 9.5709 10.4156 8.8866 ]


# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:307) nnet_out =  [
#   0.271 0.347043 0.381958 0.45458 0.45765 0.0877699 
#   0.220165 0.382573 0.397262 0.237583 0.690322 0.0720953 
#   0.334902 0.328809 0.336289 0.329809 0.350748 0.319443 
#   0.660368 0.0945027 0.245129 0.262213 0.332462 0.405325 
#   0.907799 0.00891654 0.0832847 0.177553 0.272997 0.549449 
#   0.907858 0.00891305 0.0832292 0.177892 0.272279 0.54983 ]


# VLOG[4] (nnet-train-frmshuff:Eval():nnet-loss.cc:168) virtual void kaldi::nnet1::Xent::Eval(const kaldi::VectorBase<float>&, const kaldi::CuMatrixBase<float>&, const kaldi::CuMatrixBase<float>&, kaldi::CuMatrix<float>*)

# VLOG[4] (nnet-train-frmshuff:Eval():nnet-loss.cc:169) xentropy_aux_ vector =  [
#   -1.30564 -0 -0 
#   -0 -0 -0 
#   -0 -1.11228 -0 
#   -0 -0 -0 
#   -0 -0 -2.48549 
#   -0 -0 -0 ]


# VLOG[4] (nnet-train-frmshuff:Eval():nnet-loss.cc:170) cross_entropy =  4.90341

# VLOG[4] (nnet-train-frmshuff:Eval():nnet-loss.cc:171) entropy = -0

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

# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:364) grad =  [
#   -0.729 0.347043 0.381958 0 0 0 
#   0 0 0 -0.762417 0.690322 0.0720953 
#   0.334902 -0.671191 0.336289 0 0 0 
#   0 0 0 0.262213 -0.667538 0.405325 
#   0.907799 0.00891654 -0.916715 0 0 0 
#   0 0 0 0.177892 0.272279 -0.45017 ]


# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:374) ### After 0 frames,
# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:375) ### Forward propagation buffer content :
# [0] output of <Input>  ( min -13.1945, max 12.3055, mean 0.780497, variance 67.1269, skewness -0.11369, kurtosis -1.33857 ) 
# [1] output of <AffineTransform> ( min -8.8564, max 10.0963, mean -2.40672, variance 28.9713, skewness 0.923685, kurtosis -0.0985193 ) 
# [2] output of <Sigmoid> ( min 0.000142447, max 0.999959, mean 0.299571, variance 0.162334, skewness 0.84819, kurtosis -1.07261 ) 
# [3] output of <AffineTransform> ( min -2.64279, max 3.47671, mean -0.0139365, variance 1.44231, skewness 0.959558, kurtosis 1.97216 ) 
# [4] output of <BlockSoftmax> ( min 0.00891305, max 0.907858, mean 0.333333, variance 0.045796, skewness 0.927075, kurtosis 0.887316 ) 

# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:377) ### Backward propagation buffer content :
# [0] diff of <Input>  ( min -0.170608, max 0.210615, mean 0.000929004, variance 0.0026833, skewness 0.983137, kurtosis 10.0062 ) 
# [1] diff-output of <AffineTransform> ( min -0.296259, max 0.270935, mean 0.00148713, variance 0.00957177, skewness -0.39742, kurtosis 4.98531 ) 
# [2] diff-output of <Sigmoid> ( min -2.5166, max 2.29957, mean 0.105445, variance 1.6805, skewness -0.231408, kurtosis -0.546502 ) 
# [3] diff-output of <AffineTransform> ( min -0.916715, max 0.907799, mean 0, variance 0.144116, skewness -0.472706, kurtosis 0.711057 ) 
# [4] diff-output of <BlockSoftmax> ( min -0.916715, max 0.907799, mean 0, variance 0.144116, skewness -0.472706, kurtosis 0.711057 ) 

# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:378) ### Gradient stats :
# Component 1 : <AffineTransform>, 
#   linearity_grad ( min -3.81007, max 3.67438, mean -0.088959, variance 5.9879, skewness -0.0699489, kurtosis -1.19029 ) , lr-coef 1, max-norm 0
#   bias_grad ( min -0.307456, max 0.270659, mean 0.0089228, variance 0.0571957, skewness -0.330791, kurtosis -1.5 ) , lr-coef 1
# Component 2 : <Sigmoid>, 
# Component 3 : <AffineTransform>, 
#   linearity_grad ( min -0.913346, max 0.910921, mean -3.8029e-09, variance 0.212207, skewness -0.289768, kurtosis -0.512257 ) , lr-coef 1, max-norm 0
#   bias_grad ( min -0.322313, max 0.5137, mean 2.48353e-09, variance 0.0990566, skewness 0.473875, kurtosis -1.31102 ) , lr-coef 0.1
# Component 4 : <BlockSoftmax>, 

# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:399) ### After 6 frames,
# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:400) ### Forward propagation buffer content :
# [0] output of <Input>  ( min -13.1945, max 12.3055, mean 0.780497, variance 67.1269, skewness -0.11369, kurtosis -1.33857 ) 
# [1] output of <AffineTransform> ( min -8.8564, max 10.0963, mean -2.40672, variance 28.9713, skewness 0.923685, kurtosis -0.0985193 ) 
# [2] output of <Sigmoid> ( min 0.000142447, max 0.999959, mean 0.299571, variance 0.162334, skewness 0.84819, kurtosis -1.07261 ) 
# [3] output of <AffineTransform> ( min -2.64279, max 3.47671, mean -0.0139365, variance 1.44231, skewness 0.959558, kurtosis 1.97216 ) 
# [4] output of <BlockSoftmax> ( min 0.00891305, max 0.907858, mean 0.333333, variance 0.045796, skewness 0.927075, kurtosis 0.887316 ) 

# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:402) ### Backward propagation buffer content :
# [0] diff of <Input>  ( min -0.170608, max 0.210615, mean 0.000929004, variance 0.0026833, skewness 0.983137, kurtosis 10.0062 ) 
# [1] diff-output of <AffineTransform> ( min -0.296259, max 0.270935, mean 0.00148713, variance 0.00957177, skewness -0.39742, kurtosis 4.98531 ) 
# [2] diff-output of <Sigmoid> ( min -2.5166, max 2.29957, mean 0.105445, variance 1.6805, skewness -0.231408, kurtosis -0.546502 ) 
# [3] diff-output of <AffineTransform> ( min -0.916715, max 0.907799, mean 0, variance 0.144116, skewness -0.472706, kurtosis 0.711057 ) 
# [4] diff-output of <BlockSoftmax> ( min -0.916715, max 0.907799, mean 0, variance 0.144116, skewness -0.472706, kurtosis 0.711057 ) 

# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:403) ### Gradient stats :
# Component 1 : <AffineTransform>, 
#   linearity_grad ( min -3.81007, max 3.67438, mean -0.088959, variance 5.9879, skewness -0.0699489, kurtosis -1.19029 ) , lr-coef 1, max-norm 0
#   bias_grad ( min -0.307456, max 0.270659, mean 0.0089228, variance 0.0571957, skewness -0.330791, kurtosis -1.5 ) , lr-coef 1
# Component 2 : <Sigmoid>, 
# Component 3 : <AffineTransform>, 
#   linearity_grad ( min -0.913346, max 0.910921, mean -3.8029e-09, variance 0.212207, skewness -0.289768, kurtosis -0.512257 ) , lr-coef 1, max-norm 0
#   bias_grad ( min -0.322313, max 0.5137, mean 2.48353e-09, variance 0.0990566, skewness 0.473875, kurtosis -1.31102 ) , lr-coef 0.1
# Component 4 : <BlockSoftmax>, 

# LOG (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:411) Done 1 files, 0 with no tgt_mats, 0 with other errors. [TRAINING, NOT-RANDOMIZED, 0.000191534 min, fps522.102]
# LOG (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:428) MultiTaskLoss, with 2 parallel loss functions.
# Loss 1, AvgLoss: 1.63447 (Xent), [AvgXent: 1.63447, AvgTargetEnt: 0]
# FRAME_ACCURACY >> 0% <<

# Loss 2, AvgLoss: 1.04554 (Xent), [AvgXent: 1.04554, AvgTargetEnt: 0]
# FRAME_ACCURACY >> 33.3333% <<

# Loss (OVERALL), AvgLoss: 2.68001 (MultiTaskLoss), weights 1 1 , values 1.63447 1.04554 

# LOG (nnet-train-frmshuff:PrintProfile():cu-device.cc:405) -----
# [cudevice profile]
# MulRowsVec	0.000117779s
# CuMatrixBase::CopyFromMat(from CPU)	0.000126123s
# CopyToVec	0.000145674s
# Sum	0.00021553s
# AddMatVec	0.000222921s
# AddMatMat	0.000231981s
# Set	0.000287294s
# CuMatrixBase::CopyFromMat(from other CuMatrixBase)	0.000310659s
# FindRowMaxId	0.000377893s
# CuMatrix::SetZero	0.000391483s
# CuVector::SetZero	0.000447989s
# CuMatrix::CopyToMatD2H	0.000583649s
# CheckGpuHealth	0.000668049s
# CuVector::Resize	0.000770807s
# CuMatrix::Resize	0.00158834s
# Total GPU time:	0.00739479s (may involve some double-counting)
# -----
# LOG (nnet-train-frmshuff:PrintMemoryUsage():cu-device.cc:379) Memory used: 17825792 bytes.


# # Target Interp = hard, rho = 0.2
# nnet-train-frmshuff --verbose=4 --use-gpu=yes --minibatch-size=6 --randomize=false --objective-function=multitask,xent,3,1.0,xent,3,1.0 --tgt-interp-mode=hard --rho=0.2 ark:common/feats.txt ark:exp/mtl/post1.txt exp/mtl/nnetMTL.in exp/mtl/nnetMTL.out 
# LOG (nnet-train-frmshuff:SelectGpuIdAuto():cu-device.cc:280) Selecting from 1 GPUs
# LOG (nnet-train-frmshuff:SelectGpuIdAuto():cu-device.cc:295) cudaSetDevice(0): Tesla K40c	free:11389M, used:82M, total:11471M, free/total:0.992841
# LOG (nnet-train-frmshuff:SelectGpuIdAuto():cu-device.cc:344) Trying to select device: 0 (automatically), mem_ratio: 0.992841
# LOG (nnet-train-frmshuff:SelectGpuIdAuto():cu-device.cc:363) Success selecting device 0 free mem ratio: 0.992841
# LOG (nnet-train-frmshuff:FinalizeActiveGpu():cu-device.cc:202) The active GPU is [0]: Tesla K40c	free:11372M, used:99M, total:11471M, free/total:0.991359 version 3.5
# LOG (nnet-train-frmshuff:PrintMemoryUsage():cu-device.cc:379) Memory used: 0 bytes.
# LOG (nnet-train-frmshuff:Init():nnet-randomizer.cc:31) Seeding by srand with : 777
# LOG (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:196) Objective Function = multitask,xent,3,1.0,xent,3,1.0

# LOG (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:199) TRAINING STARTED
# VLOG[3] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:211) Reading s1u1
# LOG (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:279) Finished filling randomizer. num done = 1, num_no_tgt_mat = 0, num_no_frame_wts = 0

# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:301) Mini-batch loop

# VLOG[3] (nnet-train-frmshuff:PropagateFnc():nnet/nnet-activation.h:139) Feedfwd through blocksoftmax 

# VLOG[3] (nnet-train-frmshuff:PropagateFnc():nnet/nnet-activation.h:145)  T = 1, Block = 0, num_frames =  6, num_pdf =  3

# VLOG[3] (nnet-train-frmshuff:PropagateFnc():nnet/nnet-activation.h:192) Ready to exit

# VLOG[3] (nnet-train-frmshuff:PropagateFnc():nnet/nnet-activation.h:145)  T = 1, Block = 1, num_frames =  6, num_pdf =  3

# VLOG[3] (nnet-train-frmshuff:PropagateFnc():nnet/nnet-activation.h:192) Ready to exit

# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:306) nnet_in =  [
#   -9.2396 -7.4065 -13.1945 -8.7807 -9.5492 
#   -11.8494 -10.6132 -9.5155 -4.9394 -6.0834 
#   -1.909 4.292 1.0259 -0.0892 1.0108 
#   -0.2899 -0.1756 2.1068 1.9927 2.0042 
#   10.9496 8.2924 11.0143 12.3055 10.6914 
#   11.4633 11.028 9.5709 10.4156 8.8866 ]


# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:307) nnet_out =  [
#   0.271 0.347043 0.381958 0.45458 0.45765 0.0877699 
#   0.220165 0.382573 0.397262 0.237583 0.690322 0.0720953 
#   0.334902 0.328809 0.336289 0.329809 0.350748 0.319443 
#   0.660368 0.0945027 0.245129 0.262213 0.332462 0.405325 
#   0.907799 0.00891654 0.0832847 0.177553 0.272997 0.549449 
#   0.907858 0.00891305 0.0832292 0.177892 0.272279 0.54983 ]


# VLOG[4] (nnet-train-frmshuff:Eval():nnet-loss.cc:168) virtual void kaldi::nnet1::Xent::Eval(const kaldi::VectorBase<float>&, const kaldi::CuMatrixBase<float>&, const kaldi::CuMatrixBase<float>&, kaldi::CuMatrix<float>*)

# VLOG[4] (nnet-train-frmshuff:Eval():nnet-loss.cc:169) xentropy_aux_ vector =  [
#   -0.261127 -0 -0.769957 
#   -0 -0 -0 
#   -0 -0.222456 -0.871826 
#   -0 -0 -0 
#   -0.077386 -0 -0.497098 
#   -0 -0 -0 ]


# VLOG[4] (nnet-train-frmshuff:Eval():nnet-loss.cc:170) cross_entropy =  2.69985

# VLOG[4] (nnet-train-frmshuff:Eval():nnet-loss.cc:171) entropy = -0

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

# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:364) grad =  [
#   0.0709998 0.347043 -0.418042 0 0 0 
#   0 0 -0 -0.762417 0.690322 0.0720953 
#   0.334902 0.128809 -0.463711 0 0 0 
#   -0 0 0 0.262213 -0.667538 0.405325 
#   0.107799 0.00891654 -0.116715 0 0 0 
#   0 0 0 0.177892 0.272279 -0.45017 ]


# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:374) ### After 0 frames,
# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:375) ### Forward propagation buffer content :
# [0] output of <Input>  ( min -13.1945, max 12.3055, mean 0.780497, variance 67.1269, skewness -0.11369, kurtosis -1.33857 ) 
# [1] output of <AffineTransform> ( min -8.8564, max 10.0963, mean -2.40672, variance 28.9713, skewness 0.923685, kurtosis -0.0985193 ) 
# [2] output of <Sigmoid> ( min 0.000142447, max 0.999959, mean 0.299571, variance 0.162334, skewness 0.84819, kurtosis -1.07261 ) 
# [3] output of <AffineTransform> ( min -2.64279, max 3.47671, mean -0.0139365, variance 1.44231, skewness 0.959558, kurtosis 1.97216 ) 
# [4] output of <BlockSoftmax> ( min 0.00891305, max 0.907858, mean 0.333333, variance 0.045796, skewness 0.927075, kurtosis 0.887316 ) 

# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:377) ### Backward propagation buffer content :
# [0] diff of <Input>  ( min -0.170608, max 0.210615, mean 0.00148806, variance 0.00265105, skewness 0.967926, kurtosis 10.281 ) 
# [1] diff-output of <AffineTransform> ( min -0.296259, max 0.270935, mean -0.00265987, variance 0.00925898, skewness -0.307646, kurtosis 5.45602 ) 
# [2] diff-output of <Sigmoid> ( min -2.07867, max 1.42769, mean 0.00165613, variance 0.721411, skewness -0.493283, kurtosis 0.481082 ) 
# [3] diff-output of <AffineTransform> ( min -0.762417, max 0.690322, mean -6.20882e-10, variance 0.0755402, skewness -0.646667, kurtosis 1.61156 ) 
# [4] diff-output of <BlockSoftmax> ( min -0.762417, max 0.690322, mean -6.20882e-10, variance 0.0755402, skewness -0.646667, kurtosis 1.61156 ) 

# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:378) ### Gradient stats :
# Component 1 : <AffineTransform>, 
#   linearity_grad ( min -3.10056, max 3.4864, mean 0.0857126, variance 4.30113, skewness 0.0695895, kurtosis -1.08317 ) , lr-coef 1, max-norm 0
#   bias_grad ( min -0.288578, max 0.194188, mean -0.0159592, variance 0.0407951, skewness -0.434357, kurtosis -1.5 ) , lr-coef 1
# Component 2 : <Sigmoid>, 
# Component 3 : <AffineTransform>, 
#   linearity_grad ( min -0.560372, max 0.494695, mean -5.27749e-09, variance 0.0758774, skewness -0.290482, kurtosis -0.626754 ) , lr-coef 1, max-norm 0
#   bias_grad ( min -0.998468, max 0.5137, mean 2.48353e-09, variance 0.281253, skewness -0.842182, kurtosis -0.604139 ) , lr-coef 0.1
# Component 4 : <BlockSoftmax>, 

# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:399) ### After 6 frames,
# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:400) ### Forward propagation buffer content :
# [0] output of <Input>  ( min -13.1945, max 12.3055, mean 0.780497, variance 67.1269, skewness -0.11369, kurtosis -1.33857 ) 
# [1] output of <AffineTransform> ( min -8.8564, max 10.0963, mean -2.40672, variance 28.9713, skewness 0.923685, kurtosis -0.0985193 ) 
# [2] output of <Sigmoid> ( min 0.000142447, max 0.999959, mean 0.299571, variance 0.162334, skewness 0.84819, kurtosis -1.07261 ) 
# [3] output of <AffineTransform> ( min -2.64279, max 3.47671, mean -0.0139365, variance 1.44231, skewness 0.959558, kurtosis 1.97216 ) 
# [4] output of <BlockSoftmax> ( min 0.00891305, max 0.907858, mean 0.333333, variance 0.045796, skewness 0.927075, kurtosis 0.887316 ) 

# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:402) ### Backward propagation buffer content :
# [0] diff of <Input>  ( min -0.170608, max 0.210615, mean 0.00148806, variance 0.00265105, skewness 0.967926, kurtosis 10.281 ) 
# [1] diff-output of <AffineTransform> ( min -0.296259, max 0.270935, mean -0.00265987, variance 0.00925898, skewness -0.307646, kurtosis 5.45602 ) 
# [2] diff-output of <Sigmoid> ( min -2.07867, max 1.42769, mean 0.00165613, variance 0.721411, skewness -0.493283, kurtosis 0.481082 ) 
# [3] diff-output of <AffineTransform> ( min -0.762417, max 0.690322, mean -6.20882e-10, variance 0.0755402, skewness -0.646667, kurtosis 1.61156 ) 
# [4] diff-output of <BlockSoftmax> ( min -0.762417, max 0.690322, mean -6.20882e-10, variance 0.0755402, skewness -0.646667, kurtosis 1.61156 ) 

# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:403) ### Gradient stats :
# Component 1 : <AffineTransform>, 
#   linearity_grad ( min -3.10056, max 3.4864, mean 0.0857126, variance 4.30113, skewness 0.0695895, kurtosis -1.08317 ) , lr-coef 1, max-norm 0
#   bias_grad ( min -0.288578, max 0.194188, mean -0.0159592, variance 0.0407951, skewness -0.434357, kurtosis -1.5 ) , lr-coef 1
# Component 2 : <Sigmoid>, 
# Component 3 : <AffineTransform>, 
#   linearity_grad ( min -0.560372, max 0.494695, mean -5.27749e-09, variance 0.0758774, skewness -0.290482, kurtosis -0.626754 ) , lr-coef 1, max-norm 0
#   bias_grad ( min -0.998468, max 0.5137, mean 2.48353e-09, variance 0.281253, skewness -0.842182, kurtosis -0.604139 ) , lr-coef 0.1
# Component 4 : <BlockSoftmax>, 

# LOG (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:411) Done 1 files, 0 with no tgt_mats, 0 with other errors. [TRAINING, NOT-RANDOMIZED, 0.000201833 min, fps495.459]
# LOG (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:428) MultiTaskLoss, with 2 parallel loss functions.
# Loss 1, AvgLoss: 0.89995 (Xent), [AvgXent: 0.89995, AvgTargetEnt: 0]
# FRAME_ACCURACY >> 0% <<

# Loss 2, AvgLoss: 1.04554 (Xent), [AvgXent: 1.04554, AvgTargetEnt: 0]
# FRAME_ACCURACY >> 33.3333% <<

# Loss (OVERALL), AvgLoss: 1.94549 (MultiTaskLoss), weights 1 1 , values 0.89995 1.04554 

# LOG (nnet-train-frmshuff:PrintProfile():cu-device.cc:405) -----
# [cudevice profile]
# CuArray::SetZero	0.000118494s
# CopyToVec	0.000145674s
# CuMatrixBase::CopyFromMat(from CPU)	0.000152111s
# Sum	0.000201464s
# AddMatMat	0.000227451s
# AddMatVec	0.000241518s
# CuMatrixBase::CopyFromMat(from other CuMatrixBase)	0.00031209s
# Set	0.000323296s
# CuMatrix::SetZero	0.000383854s
# FindRowMaxId	0.000409126s
# CuVector::SetZero	0.000463724s
# CuMatrix::CopyToMatD2H	0.000645876s
# CheckGpuHealth	0.000767946s
# CuVector::Resize	0.000826836s
# CuMatrix::Resize	0.00159311s
# Total GPU time:	0.00782228s (may involve some double-counting)
# -----
# LOG (nnet-train-frmshuff:PrintMemoryUsage():cu-device.cc:379) Memory used: 17825792 bytes.


# # Target Interp = soft, rho = 0.2
# nnet-train-frmshuff --verbose=4 --use-gpu=yes --minibatch-size=6 --randomize=false --objective-function=multitask,xent,3,1.0,xent,3,1.0 --tgt-interp-mode=soft --rho=0.2 ark:common/feats.txt ark:exp/mtl/post1.txt exp/mtl/nnetMTL.in exp/mtl/nnetMTL.out 
# LOG (nnet-train-frmshuff:SelectGpuIdAuto():cu-device.cc:280) Selecting from 1 GPUs
# LOG (nnet-train-frmshuff:SelectGpuIdAuto():cu-device.cc:295) cudaSetDevice(0): Tesla K40c	free:11389M, used:82M, total:11471M, free/total:0.992841
# LOG (nnet-train-frmshuff:SelectGpuIdAuto():cu-device.cc:344) Trying to select device: 0 (automatically), mem_ratio: 0.992841
# LOG (nnet-train-frmshuff:SelectGpuIdAuto():cu-device.cc:363) Success selecting device 0 free mem ratio: 0.992841
# LOG (nnet-train-frmshuff:FinalizeActiveGpu():cu-device.cc:202) The active GPU is [0]: Tesla K40c	free:11372M, used:99M, total:11471M, free/total:0.991359 version 3.5
# LOG (nnet-train-frmshuff:PrintMemoryUsage():cu-device.cc:379) Memory used: 0 bytes.
# LOG (nnet-train-frmshuff:Init():nnet-randomizer.cc:31) Seeding by srand with : 777
# LOG (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:196) Objective Function = multitask,xent,3,1.0,xent,3,1.0

# LOG (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:199) TRAINING STARTED
# VLOG[3] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:211) Reading s1u1
# LOG (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:279) Finished filling randomizer. num done = 1, num_no_tgt_mat = 0, num_no_frame_wts = 0

# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:301) Mini-batch loop

# VLOG[3] (nnet-train-frmshuff:PropagateFnc():nnet/nnet-activation.h:139) Feedfwd through blocksoftmax 

# VLOG[3] (nnet-train-frmshuff:PropagateFnc():nnet/nnet-activation.h:145)  T = 1, Block = 0, num_frames =  6, num_pdf =  3

# VLOG[3] (nnet-train-frmshuff:PropagateFnc():nnet/nnet-activation.h:192) Ready to exit

# VLOG[3] (nnet-train-frmshuff:PropagateFnc():nnet/nnet-activation.h:145)  T = 1, Block = 1, num_frames =  6, num_pdf =  3

# VLOG[3] (nnet-train-frmshuff:PropagateFnc():nnet/nnet-activation.h:192) Ready to exit

# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:306) nnet_in =  [
#   -9.2396 -7.4065 -13.1945 -8.7807 -9.5492 
#   -11.8494 -10.6132 -9.5155 -4.9394 -6.0834 
#   -1.909 4.292 1.0259 -0.0892 1.0108 
#   -0.2899 -0.1756 2.1068 1.9927 2.0042 
#   10.9496 8.2924 11.0143 12.3055 10.6914 
#   11.4633 11.028 9.5709 10.4156 8.8866 ]


# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:307) nnet_out =  [
#   0.271 0.347043 0.381958 0.45458 0.45765 0.0877699 
#   0.220165 0.382573 0.397262 0.237583 0.690322 0.0720953 
#   0.334902 0.328809 0.336289 0.329809 0.350748 0.319443 
#   0.660368 0.0945027 0.245129 0.262213 0.332462 0.405325 
#   0.907799 0.00891654 0.0832847 0.177553 0.272997 0.549449 
#   0.907858 0.00891305 0.0832292 0.177892 0.272279 0.54983 ]


# VLOG[4] (nnet-train-frmshuff:Eval():nnet-loss.cc:168) virtual void kaldi::nnet1::Xent::Eval(const kaldi::VectorBase<float>&, const kaldi::CuMatrixBase<float>&, const kaldi::CuMatrixBase<float>&, kaldi::CuMatrix<float>*)

# VLOG[4] (nnet-train-frmshuff:Eval():nnet-loss.cc:169) xentropy_aux_ vector =  [
#   -0.544189 -0.293822 -0.294091 
#   -0 -0 -0 
#   -0.293084 -0.515037 -0.293186 
#   -0 -0 -0 
#   -0.070251 -0.0336678 -0.662701 
#   -0 -0 -0 ]


# VLOG[4] (nnet-train-frmshuff:Eval():nnet-loss.cc:170) cross_entropy =  3.00003

# VLOG[4] (nnet-train-frmshuff:Eval():nnet-loss.cc:171) entropy = -0

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

# VLOG[4] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:364) grad =  [
#   -0.0987722 0.0609655 0.0378068 0 0 0 
#   0 0 0 -0.762417 0.690322 0.0720953 
#   0.0657354 -0.130631 0.0648954 0 0 0 
#   -0 0 0 0.262213 -0.667538 0.405325 
#   0.00713954 0.0330479 -0.0401874 0 0 0 
#   0 0 0 0.177892 0.272279 -0.45017 ]


# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:374) ### After 0 frames,
# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:375) ### Forward propagation buffer content :
# [0] output of <Input>  ( min -13.1945, max 12.3055, mean 0.780497, variance 67.1269, skewness -0.11369, kurtosis -1.33857 ) 
# [1] output of <AffineTransform> ( min -8.8564, max 10.0963, mean -2.40672, variance 28.9713, skewness 0.923685, kurtosis -0.0985193 ) 
# [2] output of <Sigmoid> ( min 0.000142447, max 0.999959, mean 0.299571, variance 0.162334, skewness 0.84819, kurtosis -1.07261 ) 
# [3] output of <AffineTransform> ( min -2.64279, max 3.47671, mean -0.0139365, variance 1.44231, skewness 0.959558, kurtosis 1.97216 ) 
# [4] output of <BlockSoftmax> ( min 0.00891305, max 0.907858, mean 0.333333, variance 0.045796, skewness 0.927075, kurtosis 0.887316 ) 

# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:377) ### Backward propagation buffer content :
# [0] diff of <Input>  ( min -0.170608, max 0.210615, mean 0.00111116, variance 0.00265166, skewness 0.98952, kurtosis 10.3036 ) 
# [1] diff-output of <AffineTransform> ( min -0.296259, max 0.270935, mean -0.00153868, variance 0.00926071, skewness -0.342389, kurtosis 5.46802 ) 
# [2] diff-output of <Sigmoid> ( min -2.07867, max 1.42769, mean 0.010422, variance 0.693617, skewness -0.540183, kurtosis 0.768879 ) 
# [3] diff-output of <AffineTransform> ( min -0.762417, max 0.690322, mean 4.12628e-09, variance 0.058149, skewness -0.781447, kurtosis 3.92779 ) 
# [4] diff-output of <BlockSoftmax> ( min -0.762417, max 0.690322, mean 4.12628e-09, variance 0.058149, skewness -0.781447, kurtosis 3.92779 ) 

# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:378) ### Gradient stats :
# Component 1 : <AffineTransform>, 
#   linearity_grad ( min -3.26887, max 3.51574, mean 0.0440826, variance 4.60394, skewness 0.0370405, kurtosis -1.10878 ) , lr-coef 1, max-norm 0
#   bias_grad ( min -0.292245, max 0.218626, mean -0.00923207, variance 0.0450191, skewness -0.372358, kurtosis -1.5 ) , lr-coef 1
# Component 2 : <Sigmoid>, 
# Component 3 : <AffineTransform>, 
#   linearity_grad ( min -0.560372, max 0.494695, mean 9.52019e-09, variance 0.0463935, skewness -0.308766, kurtosis 1.51937 ) , lr-coef 1, max-norm 0
#   bias_grad ( min -0.322313, max 0.295063, mean 2.42144e-08, variance 0.032935, skewness -0.211835, kurtosis -0.174352 ) , lr-coef 0.1
# Component 4 : <BlockSoftmax>, 

# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:399) ### After 6 frames,
# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:400) ### Forward propagation buffer content :
# [0] output of <Input>  ( min -13.1945, max 12.3055, mean 0.780497, variance 67.1269, skewness -0.11369, kurtosis -1.33857 ) 
# [1] output of <AffineTransform> ( min -8.8564, max 10.0963, mean -2.40672, variance 28.9713, skewness 0.923685, kurtosis -0.0985193 ) 
# [2] output of <Sigmoid> ( min 0.000142447, max 0.999959, mean 0.299571, variance 0.162334, skewness 0.84819, kurtosis -1.07261 ) 
# [3] output of <AffineTransform> ( min -2.64279, max 3.47671, mean -0.0139365, variance 1.44231, skewness 0.959558, kurtosis 1.97216 ) 
# [4] output of <BlockSoftmax> ( min 0.00891305, max 0.907858, mean 0.333333, variance 0.045796, skewness 0.927075, kurtosis 0.887316 ) 

# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:402) ### Backward propagation buffer content :
# [0] diff of <Input>  ( min -0.170608, max 0.210615, mean 0.00111116, variance 0.00265166, skewness 0.98952, kurtosis 10.3036 ) 
# [1] diff-output of <AffineTransform> ( min -0.296259, max 0.270935, mean -0.00153868, variance 0.00926071, skewness -0.342389, kurtosis 5.46802 ) 
# [2] diff-output of <Sigmoid> ( min -2.07867, max 1.42769, mean 0.010422, variance 0.693617, skewness -0.540183, kurtosis 0.768879 ) 
# [3] diff-output of <AffineTransform> ( min -0.762417, max 0.690322, mean 4.12628e-09, variance 0.058149, skewness -0.781447, kurtosis 3.92779 ) 
# [4] diff-output of <BlockSoftmax> ( min -0.762417, max 0.690322, mean 4.12628e-09, variance 0.058149, skewness -0.781447, kurtosis 3.92779 ) 

# VLOG[1] (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:403) ### Gradient stats :
# Component 1 : <AffineTransform>, 
#   linearity_grad ( min -3.26887, max 3.51574, mean 0.0440826, variance 4.60394, skewness 0.0370405, kurtosis -1.10878 ) , lr-coef 1, max-norm 0
#   bias_grad ( min -0.292245, max 0.218626, mean -0.00923207, variance 0.0450191, skewness -0.372358, kurtosis -1.5 ) , lr-coef 1
# Component 2 : <Sigmoid>, 
# Component 3 : <AffineTransform>, 
#   linearity_grad ( min -0.560372, max 0.494695, mean 9.52019e-09, variance 0.0463935, skewness -0.308766, kurtosis 1.51937 ) , lr-coef 1, max-norm 0
#   bias_grad ( min -0.322313, max 0.295063, mean 2.42144e-08, variance 0.032935, skewness -0.211835, kurtosis -0.174352 ) , lr-coef 0.1
# Component 4 : <BlockSoftmax>, 

# LOG (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:411) Done 1 files, 0 with no tgt_mats, 0 with other errors. [TRAINING, NOT-RANDOMIZED, 0.0002062 min, fps484.965]
# LOG (nnet-train-frmshuff:main():nnet-train-frmshuff.cc:428) MultiTaskLoss, with 2 parallel loss functions.
# Loss 1, AvgLoss: 1.00001 (Xent), [AvgXent: 1.00001, AvgTargetEnt: 0]
# FRAME_ACCURACY >> 0% <<

# Loss 2, AvgLoss: 1.04554 (Xent), [AvgXent: 1.04554, AvgTargetEnt: 0]
# FRAME_ACCURACY >> 33.3333% <<

# Loss (OVERALL), AvgLoss: 2.04555 (MultiTaskLoss), weights 1 1 , values 1.00001 1.04554 

# LOG (nnet-train-frmshuff:PrintProfile():cu-device.cc:405) -----
# [cudevice profile]
# Scale	0.000127077s
# CuMatrixBase::CopyFromMat(from CPU)	0.000139952s
# CopyToVec	0.000152588s
# Sum	0.00020504s
# AddMatVec	0.00022912s
# AddMatMat	0.000268459s
# Set	0.000302792s
# CuMatrixBase::CopyFromMat(from other CuMatrixBase)	0.000356913s
# FindRowMaxId	0.000396252s
# CuMatrix::SetZero	0.00041008s
# CuVector::SetZero	0.000469923s
# CuMatrix::CopyToMatD2H	0.000660419s
# CheckGpuHealth	0.000761986s
# CuVector::Resize	0.000861645s
# CuMatrix::Resize	0.00163054s
# Total GPU time:	0.00809288s (may involve some double-counting)
# -----
# LOG (nnet-train-frmshuff:PrintMemoryUsage():cu-device.cc:379) Memory used: 17825792 bytes.


