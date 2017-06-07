% This function is used to find the confusion matrix between classes.
% Inputs:
% *inSourceData:
% This is N x D matrix with N points of each D dimension. (e.g. testing 
% samples)
% *inTargetData:
% This is P x D matrix of P points of each D dimension. (e.g. P unseen 
% class prototypes)
% *inTrueLabels:
% This is N x 1 matrix containing true labels of N points in inSourceData.
% *inMetric:
% This is the metric to be used for distance claculation e.g. 'l2', 'metric'
% *inM:
% This is a metric of D x D dimensions to be used for distance calculation.
% inM is empty if inMetric is not 'metric'
% *inClassLabels:
% This is a vector of 1 X K containing K class labels
% 
% Outputs:
% *outDistanceMatrix
% This is a distance matrix of N x P.This contains the distance of each of 
% the N data points in inSourceData wrt. P datapoints in inTargetData
% *perConfusionMat
% This is confusion matrix of size K x K for K classes

function [outDistanceMatrix perConfusionMat] = funtionPlotConfusionMatrixGetDistanceMatrix(inSourceData, inTargetData, inTrueLabels, inMetric, inM, inClassLabels)

outDistanceMatrix = functionGetDistancesBetweenSamples(inSourceData, inTargetData, inMetric, inM);
[minDist, inferedClassLabels] = min(outDistanceMatrix');
inferedClassLabels = inClassLabels(inferedClassLabels);
confusionMatTrainBeforeML = confusionmat(inTrueLabels, inferedClassLabels);
perConfusionMat = confusionMatTrainBeforeML./repmat(sum(confusionMatTrainBeforeML, 2), 1, length(inClassLabels)); 

if (sum(sum(perConfusionMat, 2)) ~= length(inClassLabels))
    error('Something wrong in confusion matrix...');
end

perConfusionMat = perConfusionMat * 100;
targets = zeros(length(inClassLabels), size(inSourceData, 1));
targets = full(ind2vec(inTrueLabels'));
outputs = zeros(length(inClassLabels), size(inSourceData, 1));
outputs = full(ind2vec(inferedClassLabels));
plotconfusion(targets, outputs);


% distanceMatrix = functionGetDistancesBetweenSamples(semanticEmbeddingsTrain, attributes(:, defaultTrainClassLabels)', ...
% 'l2');
% [minDist inferedClassLabelsTrain] = min(distanceMatrix');
% confusionMatTrainBeforeML = confusionmat(labelsTrainingSubsetData, inferedClassLabelsTrain);
% perConfusionMatTrainBeforeML = confusionMatTrainBeforeML./repmat(sum(confusionMatTrainBeforeML, 2), 1, length(defaultTrainClassLabels)); 
% if (sum(sum(perConfusionMatTrainBeforeML, 2)) ~= length(defaultTrainClassLabels))
%     error('Something wrong in confusion matrix...');
% end