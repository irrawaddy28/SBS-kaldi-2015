function [loss, xent1, xent2, loss1, loss2, grad] = compute_ts_loss(mask, y1, y2, t1, t2, T, rho)
% This mimics the Kaldi function which computes the Teacher-Student loss
% void TS::Eval2(const VectorBase<BaseFloat> &frame_weights,
%                const CuMatrixBase<BaseFloat> &net_out1,
%			     const CuMatrixBase<BaseFloat> &net_out2,
%                const CuMatrixBase<BaseFloat> &target1,
%			     const CuMatrixBase<BaseFloat> &target2,
%                CuMatrix<BaseFloat> *diff)

% xent11 same as TS::Eval2::xentropy_aux1_
% xent12 same as TS::Eval2::xentropy_aux2_
% loss1 same as TS::Eval2::cross_entropy1
% loss2 same as TS::Eval2::cross_entropy2
% loss  same as TS::Eval2::cross_entropy
% grad same as TS::Eval2::(*diff)

xent1 = t1.*log(y1 + 1e-20);
xent2 = t2.*log(y2 + 1e-20);
xent1 = xent1.*mask;
xent2 = xent2.*mask;

loss1 = sum(-xent1, 2);
loss2 = sum(-xent2, 2);
loss  = rho*loss1 + (1-rho)*loss2;
grad = rho*(y1 - t1) + (1-rho)*T*(y2 - t2);
grad = grad.*mask;
