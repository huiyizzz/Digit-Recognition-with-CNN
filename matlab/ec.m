%% Network defintion
layers = get_lenet();

%% load the trained weights
load lenet.mat

%% Loading data
path = '../images/';
images = ["image1.jpg", "image2.jpg", "image3.png", "image4.jpg"];

% set padsize for each image to make the digit in the middle of the image
% The value of padsize was manually chosen for best accuracy.
pad = [50, 30, 20, 3];

for i = 1:4
    img = imread(path+images(i));
    if numel(size(img))>2
        img = rgb2gray(img); % get grayscale image
    end
    img = 255 - img; % reverse color of image
    
    % morphological opening
    se = strel('disk', 15);
    background = imopen(img, se);
    img = img - background; % form a uniform background
    img = imbinarize(img);
    img = bwareaopen(img, 5); % remove background noise
    
    if i == 4
        % processing unconnected digits
        se = strel('line', 5, 30);
        connected_img = imdilate(img, se);
        
        % in case of bounding box out of range error
        connected_img = padarray(connected_img, [1 1], 0, 'both');
        img = padarray(img, [1 1], 0, 'both');
        L = bwlabel(connected_img); % label components
    else
        L = bwlabel(img); % label components
    end

    boxes = regionprops(L, 'boundingbox'); % create bounding box
    layers{1}.batch_size = length(boxes); % set batch size
    xtest = zeros([28*28 length(boxes)]);

    figure('NumberTitle', 'off', 'Name', images(i))
    imshow(img);

    for b = 1 : length(boxes)
        % place bounding boxes
        box = boxes(b).BoundingBox;
        rect = rectangle('Position', box, 'EdgeColor', 'r');
        
        % get digit from bounding area
        digit = img(floor(box(2):box(2)+box(4)), floor(box(1):box(1)+box(3)));
        
        % pad zeros to form a square image
        [m, n] = size(digit);
        if m > n
            digit = padarray(digit, [0 round((m-n)/2)], 0, 'both');
        elseif m < n
            digit = padarray(digit, [round((n-m)/2) 0], 0, 'both');
        end
        
        % pad zeros to make the digit in the middle of the image
        digit = padarray(digit, [pad(i) pad(i)], 0, 'both');
        
        % form test set
        digit = imresize(digit, [28 28]); % resize the image
        digit = im2double(digit); % change the range and the data type
        digit = reshape(digit', [28*28 1]); % transpose and reshapes the image 
        xtest(:, b) = digit;
    end
    
    % set the actual label
    if i < 3
        ytest = [1 2 3 4 5 6 7 8 9 0] + 1;
    elseif i == 3
        ytest = [6 0 6 2 4] + 1;
    else
        ytest = [7 0 9 3 1 6 7 2 6 1 ...
                 3 9 6 4 1 4 2 0 0 5 ...
                 4 4 7 3 1 0 2 5 5 1 ...
                 7 7 4 9 1 7 4 2 9 1 ...
                 5 3 4 0 2 9 4 4 1 1] + 1;       
    end
    
    
    
    % get output from the network
    [output, P] = convnet_forward(params, layers, xtest);
    [M, ypred] = max(P);
    
    count = 0;
    for b = 1 : length(boxes)
        if ytest(b) == ypred(b)
            count = count + 1; % count the same number
        end
    end
    
    disp(images(i));
    if i == 4
        disp('Actual label:');
        disp(reshape(ytest, 5, 10));
        disp('Predicted label:');
        disp(reshape(ypred, 5, 10));
    else
        disp('Actual label:');
        disp(ytest);
        disp('Predicted label:');
        disp(ypred);
    end
    acc = count/length(boxes);
    disp('The total number of digits:');
    disp(length(boxes));
    disp('The number of correct predicted label:');
    disp(count);
    disp('The prediction accuracy for the current image:');
    disp(acc);
    if acc<1
        C = confusionmat(ytest, ypred,'order',[1,2,3,4,5,6,7,8,9,10]);
        figure
        confusionchart(C);
    end
end