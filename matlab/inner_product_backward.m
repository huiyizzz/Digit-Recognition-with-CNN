function [param_grad, input_od] = inner_product_backward(output, input, layer, param)

% Replace the following lines with your implementation.

% for this layer: hi = wi * hi-1 + bi
% dhi/dwi = hi-1; dhi/dhi-1 = wi; dhi/dbi = 1
% dl/dwi = (dl/dhi)(dhi/dwi) = (dl/dhi)(hi-1)
% dl/dbi = (dl/dhi)(dhi/dbi) = (dl/dhi)(1)
% dl/dhi-1 = (dl/dhi)(dhi/dhi-1) = (dl/dhi)(wi)

param_grad.w = (output.diff * input.data')';
param_grad.b = (output.diff * ones(1, input.batch_size)')';
input_od = (output.diff' * param.w')';

end
