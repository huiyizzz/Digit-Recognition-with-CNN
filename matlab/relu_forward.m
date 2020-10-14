function [output] = relu_forward(input)
output.height = input.height;
output.width = input.width;
output.channel = input.channel;
output.batch_size = input.batch_size;

% Replace the following line with your implementation.
output.data = input.data;

% replace all negative numbers by 0
output.data(output.data<0) = 0;

end