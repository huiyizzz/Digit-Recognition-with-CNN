function [input_od] = relu_backward(output, input, layer)

% Replace the following line with your implementation.
    % hi = max(hi-1, 0)
    % dhi/dhi-1 = 1 or 0
    % dl/dhi-1 = (dl/dhi)(dhi/dhi-1)
    input_od = output.diff;
    input_od(input.data<0) = 0;
end
