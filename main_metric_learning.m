
clc;
clear
close all;

% 1: Linux Laptop
% 2: Windows laptop
% 3: Linux Desktop
% 4: Windows Desktop
SYSTEM_PLATFORM = 1;
BASE_PATH = '';
listDatasets = {'AwA', 'Pascal-Yahoo'};
DATASET_ID = 2;
DATASET = listDatasets{DATASET_ID};
listOfKernelTypes = {'chisq', 'cosine', 'linear', 'rbf', 'rbfchisq'};
KERNEL_ID = 3;
kernelType = listOfKernelTypes{KERNEL_ID};
useKernelisedData = 1;

% listFileNamesMappedAttributes = {'awa_mappedAllAttributes', 'apy_mappedAllAttributes'};
% listFileNamesMappedAttributesLabels = {'awa_mappedAllAttributeLabels', 'apy_mappedAllAttributeLabels'};
% fileNameMappedAttributes = listFileNamesMappedAttributes{DATASET_ID};
% fileNameMappedAttributesLabels = listFileNamesMappedAttributesLabels{DATASET_ID};

%Enable/add required tool boxes
addPath = 1;
BASE_PATH = functionEnvSetup(SYSTEM_PLATFORM, addPath);
VIEW_TSNE = 1;

%% START >> Load data

inputData = functionLoadInputs(DATASET, BASE_PATH);

%% END >> Load data


%% Start >> Clustering of data
labels = zeros(1, inputData.NUMBER_OF_CLASSES);
labels(inputData.defaultTestClassLabels) = 1;
labels = 1. - labels;
defaultTrainClassLabels = find(labels);
trainClassNames = inputData.classNames(defaultTrainClassLabels);
testClassNames = inputData.classNames(inputData.defaultTestClassLabels);

%Get training class features
vggFeaturesTraining = [];
labelsTrainingData = [];
labelsTestingData = [];
indicesOfTrainingSamples = [];
indicesOfTestingSamples = [];

for classInd = 1:length(defaultTrainClassLabels)
    tmp = (inputData.datasetLabels == defaultTrainClassLabels(classInd));
    indicesOfTrainingSamples = [indicesOfTrainingSamples; find(tmp)];
    vggFeaturesTraining = [vggFeaturesTraining inputData.vggFeatures(:, find(tmp))];
    labelsTrainingData = [labelsTrainingData; defaultTrainClassLabels(classInd) * ones(sum(tmp), 1)];
end

for classInd = 1:length(inputData.defaultTestClassLabels)
    tmp = (inputData.datasetLabels == inputData.defaultTestClassLabels(classInd));
    indicesOfTestingSamples = [indicesOfTestingSamples; find(tmp)];
    labelsTestingData = [labelsTestingData; inputData.defaultTestClassLabels(classInd) * ones(sum(tmp), 1)];
end

if useKernelisedData
    %kernelData = functionGetKernel(BASE_PATH, vggFeaturesTraining', kernelType, dataset_path);
    kernelFullData = functionGetKernel(BASE_PATH, inputData.vggFeatures', kernelType, inputData.dataset_path);
    kernelTrainData = kernelFullData(indicesOfTrainingSamples, indicesOfTrainingSamples);
    kernelTestData = kernelFullData(indicesOfTestingSamples, indicesOfTrainingSamples);
end

attributesMat = [];
labelsTrainingSubsetData = [];
tempC = [];

%Prepare semantic space data by selecting fixed number of samples per class
for c_tr = 1:length(defaultTrainClassLabels)
    tmp1 = (inputData.datasetLabels == defaultTrainClassLabels(c_tr));
    col1 = find(tmp1);
    col1 = col1(1:inputData.numberOfSamplesPerTrainClass);
    tempC = [tempC; col1];
    %Prepare attribute matrix which contains attribute vec for each data
    %point in leaveOutData
    % Extract Features for each train class
    numberOfSamplesOfClass(c_tr) = inputData.numberOfSamplesPerTrainClass;%sum(leaveOutDatasetLabels==defaultTrainClassLabels(c_tr));
    attributesMat = [attributesMat; repmat(inputData.attributes(:, defaultTrainClassLabels(c_tr))', numberOfSamplesOfClass(c_tr), 1)];
    %tr_sample_ind = tr_sample_ind + tr_sample_class_ind;
    labelsTrainingSubsetData = [labelsTrainingSubsetData; defaultTrainClassLabels(c_tr) * ones(numberOfSamplesOfClass(c_tr), 1)];
    col1=[];tmp1=[];
end
indicesOfTrainingSamplesSubset = tempC;
indicesSubsetTrainDataFullTestData = [tempC; indicesOfTestingSamples];

%leaveOutData contains non-left-out training samples and all testing samples
if useKernelisedData
    regressorInputData = kernelFullData;%(indicesSubsetTrainDataFullTestData, indicesSubsetTrainDataFullTestData);
else
    %***TODO Correction here: should be vggFeatures
    regressorInputData = vggFeaturesTraining(:, tempC);
end

%Train regressor
[semanticEmbeddingsTrain mappingF semanticEmbeddingsTest]= functionTrainRegressor(regressorInputData', ...
    attributesMat, BASE_PATH, useKernelisedData, indicesOfTrainingSamplesSubset, indicesOfTestingSamples);
% save(sprintf('%s/%s.mat',dataset_path, fileNameMappedAttributes), 'mappedAllAttributes');
% save(sprintf('%s/%s.mat',dataset_path, fileNameMappedAttributesLabels), 'mappedAllAttributeLabels');

%% END >> Mapping of attributes

if 0 %VIEW_TSNE
    %Plot seen and unseen samples
%     semanticEmbeddingFullData = [semanticEmbeddingsTrain; semanticEmbeddingsTest];
%     semanticLabelsFullData = [labelsTrainingSubsetData; labelsTestingData];
%     labelsAsClassNames = functionGetLabelsAsClassNames(inputData.classNames, semanticLabelsFullData);
%     mappedData = [];
%     mappedData = funtionTSNEVisualisation(semanticEmbeddingFullData');
%     figureTitle = sprintf('Seen and Unseen class samples MAPPED');
%     functionMyScatterPlot(mappedData, labelsAsClassNames', inputData.NUMBER_OF_CLASSES, figureTitle);
%     labelsAsClassNames = [];
    
    %plot seen points
    semanticEmbeddingTrainSubsetData = semanticEmbeddingsTrain;
    semanticLabelsTrainSubsetData = labelsTrainingSubsetData;
    labelsAsClassNames = [];
    labelsAsClassNames = functionGetLabelsAsClassNames(inputData.classNames, semanticLabelsTrainSubsetData);
    mappedData = [];
    mappedData = funtionTSNEVisualisation(semanticEmbeddingTrainSubsetData');
    figureTitle = sprintf('Seen class samples MAPPED');
    functionMyScatterPlot(mappedData, labelsAsClassNames', length(trainClassNames), figureTitle);
    labelsAsClassNames = [];

    %Plot unseen class points
    semanticEmbeddingTestData = semanticEmbeddingsTest;
    semanticLabelsTestData = labelsTestingData;
    labelsAsClassNames = functionGetLabelsAsClassNames(inputData.classNames, semanticLabelsTestData);
    mappedData = [];
    mappedData = funtionTSNEVisualisation(semanticEmbeddingTestData');
    figureTitle = sprintf('Unseen class samples MAPPED');
    functionMyScatterPlot(mappedData, labelsAsClassNames', length(testClassNames), figureTitle);
    labelsAsClassNames = [];
end

%% Start >> Metric learning
%Prepare data for stochastic gradient descend
numberOfSamplesForSGDPerClass = 50;
attributesMatSubset = [];%zeros(numberOfSamplesForSGDPerClass * length(trainClassNames), size(attributesMat, 2));
attributesMatSubsetLabels = [];

for p = 1:length(trainClassNames)
    startI = inputData.numberOfSamplesPerTrainClass * (p - 1) + 1;
    endI = startI + numberOfSamplesForSGDPerClass - 1;
    attributesMatSubset = [attributesMatSubset; semanticEmbeddingsTrain(startI:endI, :)];
    attributesMatSubsetLabels = [attributesMatSubsetLabels; labelsTrainingSubsetData(startI:endI)];
end
metricLearned = functionLearnMetric(attributesMatSubset, attributesMatSubsetLabels, ...
numberOfSamplesForSGDPerClass, length(trainClassNames));
  
%% END >> Metric learning

if VIEW_TSNE
%plot seen points
    semanticEmbeddings = attributesMatSubset;
    semanticEmbeddingsLabels = attributesMatSubsetLabels;
    labelsAsClassNames = [];
    labelsAsClassNames = functionGetLabelsAsClassNames(inputData.classNames, semanticEmbeddingsLabels);
    mappedData = [];
    mappedData = funtionTSNEVisualisation(semanticEmbeddings');
    figureTitle = sprintf('Subset seen class samples MAPPED');
    functionMyScatterPlot(mappedData, labelsAsClassNames', length(trainClassNames), figureTitle);
    labelsAsClassNames = [];
    
    metricMappedAttributeSubset = attributesMatSubset * metricLearned;
    semanticEmbeddings = metricMappedAttributeSubset;
    semanticEmbeddingsLabels = attributesMatSubsetLabels;
    labelsAsClassNames = [];
    labelsAsClassNames = functionGetLabelsAsClassNames(inputData.classNames, semanticEmbeddingsLabels);
    mappedData = [];
    mappedData = funtionTSNEVisualisation(semanticEmbeddings');
    figureTitle = sprintf('Metric mapped Subset seen class samples MAPPED');
    functionMyScatterPlot(mappedData, labelsAsClassNames', length(trainClassNames), figureTitle);
    labelsAsClassNames = [];
    
end



%% START >> Semantic to semantic mapping
useKernelisedData = 0;
mappingG = [];
b = 1;
indexOfRemappedSeenPrototypes = [];

for clusterIndex = 1:numberOfClusters
    allClassesInCluster = find(ssClusteringModel.classClusterAssignment(:, 1) == clusterIndex);
    trainClassIndex = find(ismember(defaultTrainClassLabels, allClassesInCluster));
    %testClassIndex = defaultTestClassLabels; %find(ismember(defaultTestClassLabels, allClassesInCluster));
    trainClassLabels = defaultTrainClassLabels(trainClassIndex);
    indicesOfTrainClassAttributes = find(ismember(mappedAllAttributeLabels, trainClassLabels));
    remappSource = mappedAllAttributes(indicesOfTrainClassAttributes, :);
    remappTarget = attributesMat(indicesOfTrainClassAttributes, :);
    [reMappedAttributes regressor reMappedSemanticEmbeddingsTest]= functionTrainRegressor(remappSource', ...
        remappTarget, BASE_PATH, useKernelisedData, [1:size(remappSource, 1)], []);
    mappingG = [mappingG regressor];
    reMappedAllAttributesLabels = [];
    indexOfRemappedSeenPrototypes = [indexOfRemappedSeenPrototypes trainClassLabels];
    
    for m = 1:length(trainClassLabels)
        reMappedAttributesLabels = trainClassLabels(m)*ones(sum(mappedAllAttributeLabels == trainClassLabels(m)), 1);
        reMappedAllAttributesLabels = [reMappedAllAttributesLabels; reMappedAttributesLabels];
        startI = (m - 1) * numberOfSamplesPerTrainClass + 1;
        endI = (m - 1) * numberOfSamplesPerTrainClass + numberOfSamplesPerTrainClass;
        remappedSeenPrototypes(:, b) = mean(reMappedAttributes(startI:endI, :))';
        b = b + 1;
    end
    
    if VIEW_TSNE
        %         funtionTSNEVisualisation([mappedAllAttributes; reMappedAttributes]', ...
        %             [mappedAllAttributeLabelsVisualisation; reMappedAllAttributesLabels]', tmpClassLabel);
        labelsAsClassNames = functionGetLabelsAsClassNames(classNames, reMappedAllAttributesLabels);
        mappedData = [];
        mappedData = funtionTSNEVisualisation(reMappedAttributes');
        figureTitle = sprintf('Cluster %d : Seen class samples RE-MAPPED using g%d', clusterIndex, clusterIndex);
        functionMyScatterPlot(mappedData, labelsAsClassNames', length(trainClassLabels), figureTitle);
        labelsAsClassNames = [];
    end
end
%%END >> Semantic to semantic mapping


%NN
margins = [];
test_id = find(ismember(datasetLabels, defaultTestClassLabels));
for i = 1:length(test_id)
    diff = repmat(semanticEmbeddingsTest(i, :)', 1, length(defaultTestClassLabels)) - attributes(:, defaultTestClassLabels);
    nnScores = sum(diff.^2, 1)/sum(sum(diff.^2, 1));
    %     scoresAcrossClusters =  reshape(targetDomainEmbeddingsTest(:, i, :), length(defaultTrainClassLabels), numberOfClusters)'...
    %         * histogramsAllClasses(:,defaultTestClassLabels);
    margins = [margins; max(nnScores, [], 1)];
end
%%% classify
[margin id] = max(margins, [], 2);
a = (defaultTestClassLabels(id));
b = datasetLabels(test_id);
if ~sum(size(a) == size(b))
    a = a';
end
acc = 100*sum(a == b)/length(test_id)
margins = [];
meanAcc = mean(acc)
%NN

if 0
    %Training
    validClusterIndex = 1;
    validClusterIndices = [];
    histogramsUnseenClasses = functionGetSourceDomainEmbedding(defaultTestClassLabels, defaultTrainClassLabels, attributes);
    
    %Testing
    margins = [];
    test_id = find(ismember(datasetLabels, defaultTestClassLabels));
    targetDomainEmbeddingsTest = functionGetTargetDomainEmbedding(test_id, semanticEmbeddingsTest,...
        numberOfClusters, ssClusteringModel, mappingG, remappedSeenPrototypes, indexOfRemappedSeenPrototypes);
    
    for i = 1:length(test_id)
        scoresAcrossClusters =  targetDomainEmbeddingsTest(:, i)'...
            * histogramsUnseenClasses(:,defaultTestClassLabels);
        
        %     scoresAcrossClusters =  reshape(targetDomainEmbeddingsTest(:, i, :), length(defaultTrainClassLabels), numberOfClusters)'...
        %         * histogramsAllClasses(:,defaultTestClassLabels);
        margins = [margins; max(scoresAcrossClusters, [], 1)];
    end
    
    %%% classify
    [margin id] = max(margins, [], 2);
    a = (defaultTestClassLabels(id));
    b = datasetLabels(test_id);
    if ~sum(size(a) == size(b))
        a = a';
    end
    acc = 100*sum(a == b)/length(test_id)
    margins = [];
    meanAcc = mean(acc)
    %% END >> Testing
end
