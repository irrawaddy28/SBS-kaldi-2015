##  MTL Training Types
Examples of various MTL training regimes are outlined below.

### Cross-Entropy (CE) Training
```C++
nnet-train-frmshuff [options] --objective-function=<comma separated list> <feature-rspecifier> <targets-rspecifier> <model-in> [<model-out>]
nnet-train-frmshuff  --objective-function=multitask,xent,500,1.0,xent,500,1.0 scp:feature.scp  ark:posterior.ark nnet.init nnet.iter1
```
The format of --objective-function option is "multitask,\<loss1\>,\<dim1\>,\<weight1\>,...,\<lossN\>,\<dimN\>,\<weightN\>". Here, acceptable strings for \<loss\> are:

* xent (cross-entropy loss)
* ts   (teacher-student, aka knowledge distillation, loss)
* mse  (mean square error loss)

### Knowledge Distillation (KD) or Teacher-Student (TS) Training
```C++
nnet-train-frmshuff [options] --objective-function=<comma separated list> --softmax-temperature=<T>0> --rho=<[0,1]> <feature-rspecifier> <targets-rspecifier1> <targets-rspecifier2> <model-in> [<model-out>]
nnet-train-frmshuff --objective-function=multitask,ts,500,1.0,xent,500,1.0  --softmax-temperature=2 --rho=0.2  scp:feature.scp ark:1-hot-posterior.ark ark:soft-posterior.ark nnet.init nnet.iter1
```
Note that the the first loss type in --objective-function option is "ts" loss. It uses a temperature T=2 (--softmax-temperature=2) and rho=0.2 (--rho=0.2). The TS/KD loss is defined as follows:

C = rho*CE1 + (1-rho)*CE2  
where,  
CE1 = - sum_{k} d_k    log p_k(1)  
CE2 = - sum_{k} q_k(T) log p_k(T)   
d_k = ground truth labels (aka 1-hot labels) from the corpus,  
p_k(T) = softmax output of the student network parameterized by T  
                       = exp(z_k / T)/ ( sum_j exp(z_j / T) )  (z = logits of the student nnet)  
q_k(T) = softmax output of the teacher network parameterized by T (aka soft labels)  
          = exp(v_k / T)/ ( sum_j exp(v_j / T) )  (v = logits of the teacher nnet)  

For examples, see 
```
src/debug/teacher-student/train-teacherstudent-dnn.sh
src/debug/teacher-student/train-teacherstudent-mtl.sh
```
          
### Target Interpolation (TI) Training
```C++
nnet-train-frmshuff [options] --objective-function=<comma separated list> --tgt-interp-mode=<none|soft|hard> --rho=<[0,1]> <feature-rspecifier> <targets-rspecifier> <model-in> [<model-out>]
nnet-train-frmshuff  --objective-function=multitask,xent,500,1.0,xent,500,1.0   --tgt-interp-mode=soft --rho=0.4 scp:feature.scp  ark:posterior.ark nnet.init nnet.iter1
```
The CE loss when TI is enabled is defined as follows:

C = rho*CE1 + (1-rho)*CE2  
where,  
CE1 = - sum_{k} d_k  log p_k  
CE2 = - sum_{k} p_k  log p_k   (when --tgt-interp-mode=soft)  
CE2 = - sum_{k} 1_[k = argmax p]  log p_k   (when --tgt-interp-mode=hard)   
CE2 = 0 (when --tgt-interp-mode=none)    
d_k = ground truth labels (aka 1-hot labels) from the corpus,   
p_k = softmax output of the DNN                    
          
For examples, see 
```
src/debug/teacher-student/train-targetinterpolated-mtl.sh
```

