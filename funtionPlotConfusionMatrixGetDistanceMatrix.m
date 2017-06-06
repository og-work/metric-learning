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