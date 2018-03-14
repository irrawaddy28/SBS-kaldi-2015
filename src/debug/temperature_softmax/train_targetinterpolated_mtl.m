function train_targetinterpolated_mtl(tgt_interp_mode, rho)
% Usage:
% train_targetinterpolated_mtl('none',1)
% train_targetinterpolated_mtl('hard',0.2)
% train_targetinterpolated_mtl('soft',0.2)

% input features, one feature vector per row
% frames 1,3,5 belong to task 1, frames 2,4,6 belong to task 2
in=[-9.2396 -7.4065 -13.1945 -8.7807 -9.5492 
  -11.8494 -10.6132 -9.5155 -4.9394 -6.0834 
  -1.909 4.292 1.0259 -0.0892 1.0108 
  -0.2899 -0.1756 2.1068 1.9927 2.0042 
  10.9496 8.2924 11.0143 12.3055 10.6914 
  11.4633 11.028 9.5709 10.4156 8.8866 ];

% Create a matrix "block" of size #num tasks x 2 where each row contains the index
% of the start and end nodes of a block softmax
% [start node of softmax 1      end node of softmax 1]
%  start node of softmax 2      end node of softmax 2
%  ...
%  start node of softmax N      end node of softmax N
blocks = [1 3;
          4 6];
nTasks = size(blocks, 1);

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
% Same as MultiTaskLoss::tgt_mat_
t1 =  [1 0 0 0 0 0
       0 0 0 1 0 0 
       0 1 0 0 0 0
       0 0 0 0 1 0 
       0 0 1 0 0 0
       0 0 0 0 0 1];

% Masks for MTL (same as MultiTaskLoss::Eval::frmwei_have_tgt)
mask = [1 1 1 0 0 0
        0 0 0 1 1 1
        1 1 1 0 0 0
        0 0 0 1 1 1
        1 1 1 0 0 0
        0 0 0 1 1 1];
    
logits = sigmoid(in*W1' + b1)*W2' + b2;
y1 = ApplyBlockSoftmaxPerRow(logits, blocks, 1.0); % MultiTaskLoss::Eval::net_out

% Replicate MultiTaskLoss::Eval() function
for j=1:nTasks
    start_node = blocks(j,1);
    end_node   = blocks(j,2);
    active_nodes = start_node:end_node;
    if j == 1
      [ce1, xent1, grad1] = compute_xent_loss(mask(:,active_nodes), y1(:,active_nodes), t1(:,active_nodes), tgt_interp_mode, rho);
    else
      [ce2, xent2, grad2] = compute_xent_loss(mask(:,active_nodes), y1(:,active_nodes), t1(:,active_nodes), 'none');
    end
end
grad = [grad1 grad2];

y1
t1
ce1
sum(ce1)
xent1
ce2
sum(ce2)
xent2
grad


function y= sigmoid(x)
y = 1./(1+exp(-x));