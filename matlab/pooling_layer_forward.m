function [output] = pooling_layer_forward(input, layer)

    h_in = input.height;
    w_in = input.width;
    c = input.channel;
    batch_size = input.batch_size;
    k = layer.k;
    pad = layer.pad;
    stride = layer.stride;
    
    h_out = (h_in + 2*pad - k) / stride + 1;
    w_out = (w_in + 2*pad - k) / stride + 1;
    
    output.height = h_out;
    output.width = w_out;
    output.channel = c;
    output.batch_size = batch_size;

    % Replace the following line with your implementation.
    output.data = zeros([h_out, w_out, c, batch_size]);
    data = reshape(input.data, [h_in, w_in, c, batch_size]);
    data = padarray(data, [pad pad], 0, 'both');
    
    % form each kernel and find the max value in the kernel
    for i = 1:h_out
        row = (i-1) * stride;
        for j = 1:w_out
             col = (j-1) * stride;
             kernel = data(row+1:row+k, col+1:col+k, :, :);
             output.data(i, j, :, :) = max(max(kernel));
        end
    end

    output.data = reshape(output.data, [h_out*w_out*c, batch_size]);
end
