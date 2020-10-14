%% Network defintion
layers = get_lenet();

%% load the trained weights
load lenet.mat

%% Loading data
path = '../real_world_images/';
images = dir([path '*.png']);
ytest = [0, 1, 2, 3, 5, 8] + 1; % set the actual label

% set batch_size = 1
layers{1}.batch_size = 1;

for i = 1:length(images)
    img = imread([path images(i).name]);
    
    if numel(size(img))>2
        img = rgb2gray(img); % get grayscale image
    end
   
    img = imresize(img, [28 28]); % resize the image
    img = im2double(img); % change the range and the data type
    
    % get hint from piazza question @24
    img = reshape(img', [28*28 1]); % transpose and reshapes the image
    
    % real-world testing
    [output, P] = convnet_forward(params, layers, img);
    
    % get the largest value in each column with their row index (label)
    [M, ypred] = max(P);
    
    %show results
    disp('Actual label:')
    disp(ytest(i));
    disp('Predicted label: ');
    disp(ypred);
    
end