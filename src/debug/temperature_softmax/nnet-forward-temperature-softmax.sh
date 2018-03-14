#!/bin/bash

. ./path.sh

utils=../../../egs/wsj/s5/utils
feats=common/feats.txt

nnet_proto=proto1L
nnet_init=nnet1L.in

$utils/nnet/make_nnet_proto.py 5 3 0 1000 > $nnet_proto

nnet-initialize --binary=false $nnet_proto $nnet_init

echo "Features"
cat $feats

echo "Nnet"
cat $nnet_init

nnet-forward --softmax-temperature=2.0 $nnet_init ark:$feats ark,t:-

rm $nnet_proto $nnet_init

# Expected output
# > ./nnet-forward-temperature-softmax.sh 

# nnet-initialize --binary=false proto1L nnet1L.in 
# VLOG[1] (nnet-initialize:Init():nnet-nnet.cc:378) <NnetProto>
# VLOG[1] (nnet-initialize:Init():nnet-nnet.cc:378) <AffineTransform> <InputDim> 5 <OutputDim> 3 <BiasMean> 0.000000 <BiasRange> 0.000000 <ParamStddev> 1.750000
# VLOG[1] (nnet-initialize:Init():nnet-nnet.cc:378) <Softmax> <InputDim> 3 <OutputDim> 3
# VLOG[1] (nnet-initialize:Init():nnet-nnet.cc:378) </NnetProto>
# LOG (nnet-initialize:main():nnet-initialize.cc:64) Written initialized model to nnet1L.in
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
# nnet-forward --softmax-temperature=2.0 nnet1L.in ark:common/feats.txt ark,t:- 
# LOG (nnet-forward:SelectGpuId():cu-device.cc:83) Manually selected to compute on CPU.
# s1u1  [
#   0.9251708 0.07482918 2.667534e-11 
#   0.06967638 0.9303236 2.187342e-08 
#   0.04174824 0.9485873 0.009664388 
#   0.008888842 0.05601443 0.9350967 
#   8.469895e-12 3.766838e-13 1 
#   1.168306e-09 1.494102e-11 1 ]
# LOG (nnet-forward:main():nnet-forward.cc:221) Done 1 files in 3.03189e-06min, (fps 33332.2)
