%% Network defintion
layers = get_lenet();

%% Loading data
fullset = false;
[xtrain, ytrain, xvalidate, yvalidate, xtest, ytest] = load_mnist(fullset);

% load the trained weights
load lenet.mat

%% Testing the network
% Modify the code to get the confusion matrix
ypred = zeros(1, size(xtest, 2));
for i=1:100:size(xtest, 2)
    [output, P] = convnet_forward(params, layers, xtest(:, i:i+99));
    
    % get the largest value in each column with their row index (label)
    [M, ypred(i:i+99)] = max(P);
end
% use confusionmat() to get confusion matrix
C = confusionmat(ytest, ypred);
confusionchart(C)
disp('The confusion matrix is')
disp(C)