function train_teacherstudent_mtl(T, rho)

% input features, one feature vector per row
% frames 1,3,5 belong to task 1, frames 2,4,6 belong to task 2
in=[-9.2396 -7.4065 -13.1945 -8.7807 -9.5492 
  -11.8494 -10.6132 -9.5155 -4.9394 -6.0834 
  -1.909 4.292 1.0259 -0.0892 1.0108 
  -0.2899 -0.1756 2.1068 1.9927 2.0042 
  10.9496 8.2924 11.0143 12.3055 10.6914 
  11.4633 11.028 9.5709 10.4156 8.8866 ];
n_frames = size(in, 1);

% Create a matrix "block" of size #num tasks x 2 where each row contains the index
% of the start and end nodes of a block softmax
% [start node of softmax 1      end node of softmax 1]
%  start node of softmax 2      end node of softmax 2
%  ...
%  start node of softmax N      end node of softmax N
blocks = [1 3;
          4 6];
nTasks = size(blocks, 1);

% Teacher Nnet weights
W_teacher=[2.068715 -2.514695 -0.912265 1.04992745 1.04172325 
  2.07870442 2.449972 -0.4911808 -0.5379869 -0.3142128 
  2.649026 -2.479693 -0.5505048 2.179074 0.1373862 ];

% Student Nnet weights
W1 = [
  0.5971861 -0.1485795 -0.8406976 0.0259598 0.0149312 
  0.02271999 -0.1298956 -0.1417916 -0.1553033 -0.09070534 
  0.7647071 -0.4271501 -0.1589169 0.6290438 0.03965995 ];
b1 = [-3.203336 -2.703211 -1.149163];
W2 = [0.9878805 -0.4803099 3.474305 
  0.3067766 0.5243074 -1.148489 
  0.3908977 0.5411048 1.085848 
  -1.024073 -0.02004626 -1.370618 
  -3.096377 2.201577 -0.9385669 
  -1.367471 -1.489618 -0.239988 ];
b2 = [0 0 0 0 0 0];

% Hard Targets: col 1-3 for task 1, col 4-6 for task 2
% Same as MultiTaskLoss::Eval2::tgt_mat1_
t1 =  [1 0 0 0 0 0
       0 0 0 1 0 0 
       0 1 0 0 0 0
       0 0 0 0 1 0 
       0 0 1 0 0 0
       0 0 0 0 0 1];

% Soft Targets: col 1-3 for task 1, col 4-6 for task 2
% Same as MultiTaskLoss::Eval2::tgt_mat2_
% t2 = [0.9966678 0.0004233406 0.002908835 0 0 0;
%       0 0 0 0 0 0;       
%       3.549149e-05 0.9999485 1.593716e-05 0 0 0;
%       0 0 0 0 0 0;
%       0.000484859 0.1841597 0.8153554 0 0 0;       
%       0 0 0 0 0 0];
t2 = ApplyBlockSoftmaxPerRow(in*W_teacher', blocks(1,:), T);
t2 = [t2 zeros(n_frames, 3)];

% Masks for MTL (same as MultiTaskLoss::Eval2::frmwei_have_tgt)
mask = [1 1 1 0 0 0
        0 0 0 1 1 1
        1 1 1 0 0 0
        0 0 0 1 1 1
        1 1 1 0 0 0
        0 0 0 1 1 1];
    
logits = sigmoid(in*W1' + b1)*W2' + b2;
y1 = ApplyBlockSoftmaxPerRow(logits, blocks, 1.0); % MultiTaskLoss::Eval2::net_out1 
y2 = ApplyBlockSoftmaxPerRow(logits, blocks, T);   % MultiTaskLoss::Eval2::net_out2

% Replicate MultiTaskLoss::Eval2() function
for j=1:nTasks
    start_node = blocks(j,1); 
    end_node = blocks(j,2);
    active_nodes = start_node:end_node;
    if j == 1
      [ce1, xent11, xent12, ce11, ce12, grad1] = compute_ts_loss(mask(:,active_nodes), ....
          y1(:,active_nodes), y2(:,active_nodes), t1(:,active_nodes), t2(:,active_nodes), ...
          T, rho);
    else 
      [ce2, xent2, grad2] = compute_xent_loss(mask(:,active_nodes), y1(:,active_nodes), t1(:,active_nodes));
    end
end
grad = [grad1 grad2];

y1
t1
y2
t2
xent11
ce11
xent12
ce12
ce1
xent2
ce2
grad

function y= sigmoid(x)
y = 1./(1+exp(-x));