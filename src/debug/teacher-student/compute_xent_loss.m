function [loss, xent, grad] = compute_xent_loss(mask, y, t, varargin)
% This mimics the Kaldi function which computes the cross-entropy loss
% void Xent::Eval(const VectorBase<BaseFloat> &frame_weights,
%                 const CuMatrixBase<BaseFloat> &net_out, 
%                 const CuMatrixBase<BaseFloat> &target, 
%                 CuMatrix<BaseFloat> *diff) {

% loss  same as Xent::Eval::cross_entropy
% xent  same as Xent::Eval::xentropy_aux_
% grad  same as Xent::Eval::(*diff)

narginchk(3, inf);
epsilon = 1e-20;

tgt_interp_mode = 'none';
rho = 1;
if nargin > 3
  tgt_interp_mode = varargin{1};
end
if nargin > 4
  rho = varargin{2};
end

[nframes, npdfs] = size(y);
d = t; % d = ground truth targets
% Soft Targets: col 1-3 for task 1, col 4-6 for task 2
if strcmpi(tgt_interp_mode,'none')  
  rho = 1;
  t2 = zeros(size(d));
  fprintf(1, 'Overriding rho; setting it to 1 since no interpolation\n');
elseif strcmpi(tgt_interp_mode,'hard')
  [~, m_ind] = max(y,[],2);
  t2 = double((1:npdfs == m_ind));  
elseif strcmpi(tgt_interp_mode,'soft')
  t2 = y;  
else
  error('Target interpolation mode %s not supported', tgt_interp_mode);
end

t = rho*d + (1-rho)*t2;
xent = t.*log(y + epsilon); 
xent = xent.*mask;

loss =  sum(-xent, 2);
if strcmpi(tgt_interp_mode,'none') ||  strcmpi(tgt_interp_mode,'hard')
  grad =  (y - t);
  grad = mask.*grad;
elseif strcmpi(tgt_interp_mode,'soft')
  I = -log(y + epsilon);
  H = sum(y.*I, 2);
  temp = bsxfun(@minus, I, H);
  grad = y.*(rho + (1-rho)*temp) - rho*d;
  grad = mask.*grad;
else
  error('Target interpolation mode %s not supported', tgt_interp_mode);
end
  