function train-teacherstudent-dnn(T, rho)

% input features, one feature vector per row
in=[-9.2396 -7.4065 -13.1945 -8.7807 -9.5492 
  -11.8494 -10.6132 -9.5155 -4.9394 -6.0834 
  -1.909 4.292 1.0259 -0.0892 1.0108 
  -0.2899 -0.1756 2.1068 1.9927 2.0042 
  10.9496 8.2924 11.0143 12.3055 10.6914 
  11.4633 11.028 9.5709 10.4156 8.8866 ];

% Nnet weights
%W= [3.068715 -1.514695 -1.912265 0.04992745 0.04172325 
%  0.07870442 -0.449972 -0.4911808 -0.5379869 -0.3142128 
%  2.649026 -1.479693 -0.5505048 2.179074 0.1373862];
W1 = [
  0.5971861 -0.1485795 -0.8406976 0.0259598 0.0149312 
  0.02271999 -0.1298956 -0.1417916 -0.1553033 -0.09070534 
  0.7647071 -0.4271501 -0.1589169 0.6290438 0.03965995 ];
W2 = [1.209901 -0.5882571 4.255136 
  0.375723 0.6421427 -1.406606 
  0.4787499 0.6627153 1.329886 ];

% Hard Targets
t1 =  [1 0 0 
      1 0 0 
      0 1 0 
      0 1 0 
      0 0 1 
      0 0 1];
% Soft Targets
t2 = [0.9966678 0.0004233406 0.002908835;
      0.995427 1.428474e-09 0.004572987;       
      3.549149e-05 0.9999485 1.593716e-05;
      0.3672549 0.01951881 0.6132263
      0.000484859 0.1841597 0.8153554;       
      2.623838e-07 0.9996778 0.0003219858];

% out = in*W'; % logits
out = sigmoid(in*W1')*W2'; % logits
y1 = ApplySoftmaxPerRow(out, 1.0);   
y2 = ApplySoftmaxPerRow(out, T);

xentropy_aux1 = t1.*log(y1 + 1e-20);
xentropy_aux2 = (T^2)*t2.*log(y2 + 1e-20);

cross_entropy1 = sum(-xentropy_aux1, 2);
cross_entropy2 = sum(-xentropy_aux2, 2);
cross_entropy  = rho*cross_entropy1 + (1-rho)*cross_entropy2;
grad = rho*(y1 - t1) + (1-rho)*T*(y2 - t2);

y1
t1
y2
t2
xentropy_aux1
cross_entropy1
xentropy_aux2
cross_entropy2
cross_entropy
grad

function y= sigmoid(x)
y = 1./(1+exp(-x));