
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

listNormalisationTypes = {'none', 'l1', 'l2', 'metric'};
normTypeIndex = 1;
normType = listNormalisationTypes{normTypeIndex};

listSemanticSpace = {'attributes', 'word2vec'};
SEMANTIC_SPACE_ID = 1;
semanticSpace = listSemanticSpace{SEMANTIC_SPACE_ID};

%Enable/add required tool boxes
addPath = 1;
VIEW_TSNE = 0;
useKernelisedData = 1;

BASE_PATH = functionEnvSetup(SYSTEM_PLATFORM, addPath);

%% START >> Load data
inputData = functionLoadInputs(DATASET, BASE_PATH, semanticSpace);
%Normalise attributes feature
attributes = functionNormaliseData(inputData.attributes', normType);
vggFeatures =  functionNormaliseData(inputData.vggFeatures', normType);
%% END >> Load data


%% Start >> Mapping of attributes
labels = zeros(1, inputData.NUMBER_OF_CLASSES);
labels(inputData.defaultTestClassLabels) = 1;
labels = 1. - labels;
defaultTrainClassLabels = find(labels);
trainClassNames = inputData.classNames(defaultTrainClassLabels);
testClassNames = inputData.classNames(inputData.defaultTestClassLabels);

%Get training class features
vggFeaturesTraining = [];
vggFeaturesTesting = [];
labelsTrainingData = [];
labelsTestingData = [];
indicesOfTrainingSamples = [];
indicesOfTestingSamples = [];

for classInd = 1:length(defaultTrainClassLabels)
    tmp = [];
    tmp = (inputData.datasetLabels == defaultTrainClassLabels(classInd));
    indicesOfTrainingSamples = [indicesOfTrainingSamples; find(tmp)];
    vggFeaturesTraining = [vggFeaturesTraining vggFeatures(:, find(tmp))];
    labelsTrainingData = [labelsTrainingData; defaultTrainClassLabels(classInd) * ones(sum(tmp), 1)];
end

for classInd = 1:length(inputData.defaultTestClassLabels)
    tmp = [];
    tmp = (inputData.datasetLabels == inputData.defaultTestClassLabels(classInd));
    indicesOfTestingSamples = [indicesOfTestingSamples; find(tmp)];
    vggFeaturesTesting = [vggFeaturesTesting, vggFeatures(:, find(tmp))];
    labelsTestingData = [labelsTestingData; inputData.defaultTestClassLabels(classInd) * ones(sum(tmp), 1)];
end

if useKernelisedData
    normalise = 0;
    kernelFullData = functionGetKernel(BASE_PATH, vggFeatures', kernelType, inputData.dataset_path, ...
        DATASET, normalise);
    kernelTrainData = kernelFullData(indicesOfTrainingSamples, indicesOfTrainingSamples);
    kernelTestData = kernelFullData(indicesOfTestingSamples, indicesOfTrainingSamples);
end

attributesMat = [];
labelsTrainingSubsetData = [];
tempC = [];

%Prepare visual features data by selecting fixed number of samples per class
for c_tr = 1:length(defaultTrainClassLabels)
    tmp1 = (inputData.datasetLabels == defaultTrainClassLabels(c_tr));
    col1 = find(tmp1);
    col1 = col1(1:inputData.numberOfSamplesPerTrainClass);
    tempC = [tempC; col1];
    %Prepare attribute matrix which contains attribute vec for each data
    numberOfSamplesOfClass(c_tr) = inputData.numberOfSamplesPerTrainClass;%sum(leaveOutDatasetLabels==defaultTrainClassLabels(c_tr));
    attributesMat = [attributesMat; repmat(attributes(:, defaultTrainClassLabels(c_tr))', numberOfSamplesOfClass(c_tr), 1)];
    labelsTrainingSubsetData = [labelsTrainingSubsetData; defaultTrainClassLabels(c_tr) * ones(numberOfSamplesOfClass(c_tr), 1)];
    col1=[];tmp1=[];
end

indicesOfTrainingSamplesSubset = tempC;
indicesSubsetTrainDataFullTestData = [tempC; indicesOfTestingSamples];

if useKernelisedData
    regressorInputData = kernelFullData;%(indicesSubsetTrainDataFullTestData, indicesSubsetTrainDataFullTestData);
else
    %***TODO Correction here: should be vggFeatures
    regressorInputData = vggFeaturesTraining(:, tempC);
end

%% Start >> Train regressor
if 1
    [semanticEmbeddingsTrain mappingF semanticEmbeddingsTest]= functionTrainRegressor(regressorInputData', ...
        attributesMat, BASE_PATH, useKernelisedData, indicesOfTrainingSamplesSubset, indicesOfTestingSamples);
    regressorData.semanticEmbeddingsTrain = semanticEmbeddingsTrain;
    regressorData.semanticEmbeddingsTest = semanticEmbeddingsTest;
    save(sprintf('%s-regressor-vgg-%s-%s.mat', DATASET, semanticSpace, kernelType), 'regressorData');
else
    regD = load(sprintf('%s-regressor-vgg-%s-%s.mat', DATASET, semanticSpace, kernelType));
    semanticEmbeddingsTrain = regD.regressorData.semanticEmbeddingsTrain;
    semanticEmbeddingsTest = regD.regressorData.semanticEmbeddingsTest;
end
% semanticEmbeddingsTest(semanticEmbeddingsTest < 0) = 0;
% semanticEmbeddingsTrain(semanticEmbeddingsTrain < 0) = 0;
%% END >> train regressor

%% START >> Clustering of semantic embeddings train

%% END >> Clusteirng of semantic embeddings train


%% END >> Mapping of attributes
% figure;
% distanceMatrix = funtionPlotConfusionMatrixGetDistanceMatrix(semanticEmbeddingsTrain, attributes(:, defaultTrainClassLabels)', ...
%     labelsTrainingSubsetData, 'l2', [], defaultTrainClassLabels);
% title('Training data');

% figure;
% [distanceMatrixTest perConfusionMatTest]= funtionPlotConfusionMatrixGetDistanceMatrix(semanticEmbeddingsTest, attributes(:, inputData.defaultTestClassLabels)', ...
%     labelsTestingData, 'l2', [], inputData.defaultTestClassLabels);
% title('Without metric learning');

%Accuracy on Test Data
[accuracyTestWithoutML inferedLabels classDistances] = functionGetAccuracyOnDataset(semanticEmbeddingsTest, labelsTestingData, ...
    [inputData.defaultTestClassLabels], attributes, eye(size(semanticEmbeddingsTest, 2)));
% = confusionmat(labelsTestingData, inferedLabels);
accuracyTestWithoutML
inferedLabels = [];classDistances = [];

% distanceMatrix = pdist2(semanticEmbeddingsTrain, semanticEmbeddingsTrain, 'euclidean');

if VIEW_TSNE
    funtionVisualiseData(vggFeatures(:, tempC)', ...
        labelsTrainingSubsetData, inputData.classNames, inputData.NUMBER_OF_CLASSES, ...
        'VGG features training');
    
    %Plot seen and unseen samples
    funtionVisualiseData([semanticEmbeddingsTrain; semanticEmbeddingsTest], ...
        [labelsTrainingSubsetData; labelsTestingData], inputData.classNames, inputData.NUMBER_OF_CLASSES, ...
        'Seen and Unseen class samples MAPPED');
    
    % %Prototypes
    funtionVisualiseData(attributes', ...
        [1:inputData.NUMBER_OF_CLASSES], inputData.classNames, inputData.NUMBER_OF_CLASSES, ...
        'Attributes (prototypes) seen and unseen classes');
    
    %plot seen points
    funtionVisualiseData(semanticEmbeddingsTrain, ...
        labelsTrainingSubsetData, inputData.classNames, length(trainClassNames), ...
        'Seen class samples MAPPED');
    
    %Plot unseen class points with unseen prototypes
    unseenPrototypes = attributes(:, inputData.defaultTestClassLabels);
    unseenPrototypesLabels = [inputData.NUMBER_OF_CLASSES + 1 : inputData.NUMBER_OF_CLASSES + length(inputData.defaultTestClassLabels)];
    ext = {'21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32'};
    classNamesExtended = [inputData.classNames; ext'];
    funtionVisualiseData([semanticEmbeddingsTest; unseenPrototypes'], ...
        [labelsTestingData; unseenPrototypesLabels'], classNamesExtended, 2*length(testClassNames), ...
        'Embedded Unseen class samples and unseen prototypes');
end

%% Start >> Metric learning
indices = 1:ceil(0.7*length(defaultTrainClassLabels));
trainClasses = defaultTrainClassLabels(indices);
heldOutSeenClasses = defaultTrainClassLabels(indices(length(indices)) + 1: length(defaultTrainClassLabels));

%Validation Data for metric learning
%numberOfPerClassSamplesForValidation = inputData.numberOfSamplesPerTrainClass - numberOfSamplesForSGDPerClass;
numberOfPerClassSamplesForValidation = inputData.numberOfSamplesPerTrainClass;
attributesMatValidation = [];
attributesMatValidationLabels = [];

for p = heldOutSeenClasses%length(defaultTrainClassLabels)%length(trainClassNames)
    startI = inputData.numberOfSamplesPerTrainClass * (p - 1) + 1;
    endI = startI + numberOfPerClassSamplesForValidation - 1;
    attributesMatValidation = [attributesMatValidation; semanticEmbeddingsTrain(startI:endI, :)];
    attributesMatValidationLabels = [attributesMatValidationLabels; labelsTrainingSubsetData(startI:endI)];
end
    
%Prepare data for stochastic gradient descend
numberOfSamplesForSGDPerClass = 1;
randomItrMax = 1;
randomSampleIndices = [];

%Generate tuples of random samples
% for b = 1:numberOfSamplesForSGDPerClass
%     randomSampleIndices(:, b) = randi([1 inputData.numberOfSamplesPerTrainClass], randomItrMax, 1);
% end

%randomSampleIndices = [1 2];
tic
handleW = waitbar(0,'Random sampling in progress...');
arrayClassAccu = [];

for randSmpleItr = 1:randomItrMax
    
    attributesMatSubset = [];%zeros(numberOfSamplesForSGDPerClass * length(trainClassNames), size(attributesMat, 2));
    attributesMatSubsetLabels = [];
    
    %Train data for metric learning
    %     for p = 1:length(defaultTrainClassLabels) %tmpTrainClasses%
    %         startI = inputData.numberOfSamplesPerTrainClass * (p - 1);
    %         sampleIndices = startI + randomSampleIndices(randSmpleItr, :);
    %         attributesMatSubset = [attributesMatSubset; semanticEmbeddingsTrain(sampleIndices, :)];
    %         attributesMatSubsetLabels = [attributesMatSubsetLabels; labelsTrainingSubsetData(sampleIndices)];
    %     end
    
    %Use only seen class prototypes for metric learning
    attributesMatSubset = attributes(:, trainClasses)';
    attributesMatSubsetLabels = trainClasses';%defaultTrainClassLabels';
    
    %Use seen and unseen class prototypes
%     attributesMatSubset = attributes';
%     attributesMatSubsetLabels = [defaultTrainClassLabels inputData.defaultTestClassLabels]';
    
    if 0
        %Use unseen class prototypes
        unseenProto = [];unseenProtoLabels=[];
        for h = 1:length(inputData.defaultTestClassLabels)
            unseenProto = [unseenProto; repmat(attributes(:, inputData.defaultTestClassLabels(h)), 1, numberOfSamplesForSGDPerClass)'];
            unseenProtoLabels = [unseenProtoLabels; repmat(inputData.defaultTestClassLabels(h), 1, numberOfSamplesForSGDPerClass)'];
        end
        
        attributesMatSubset = [attributesMatSubset; unseenProto];
        attributesMatSubsetLabels = [attributesMatSubsetLabels; unseenProtoLabels];
    end
    
    %Prepare data for metric learning
    inputDataMetricLearn.data = attributesMatSubset;
    inputDataMetricLearn.labels = attributesMatSubsetLabels;
    inputDataMetricLearn.numberOfSamplesPerClass = numberOfSamplesForSGDPerClass;
    inputDataMetricLearn.numberOfClasses = length(trainClasses);
    lambdaArray = 0.1:0.2:2;
    marginArray = 1:100:1000;
    maxIterationsArray = 100:100:500;
    maxCount = length(lambdaArray)*length(marginArray)*length(maxIterationsArray);
    loss = {};
    metricLearned = {};
    m = 1;
    
    for lambda = lambdaArray
        for margin = marginArray
            for maxIterations = maxIterationsArray
                waitbar(m/maxCount);
                inputDataMetricLearn.lambda = lambda;
                inputDataMetricLearn.margin = margin;
                inputDataMetricLearn.maxIterations = maxIterations;
                outMLData = functionLearnMetric(inputDataMetricLearn);
                [eigVec eigVal] = eig((outMLData.metricLearned + outMLData.metricLearned')/2);
                
                if VIEW_TSNE
                    %plot seen points
                    funtionVisualiseData(attributesMatSubset, ...
                        attributesMatSubsetLabels, inputData.classNames, length(trainClassNames), ...
                        'Subset seen class samples MAPPED');
                end
                
                %figure;
                %     [distanceMatrixTest perConfusionMatTestAfterML]= funtionPlotConfusionMatrixGetDistanceMatrix(semanticEmbeddingsTest, attributes(:, inputData.defaultTestClassLabels)', ...
                %         labelsTestingData, 'metric', metricLearned, inputData.defaultTestClassLabels);
                %     title('With metric learning');
                
                %Accuracy on train data
                [accuracyTrain inferedLabels classDistances] = functionGetAccuracyOnDataset(attributesMatSubset, attributesMatSubsetLabels, ...
                    trainClasses, attributes, outMLData.metricLearned);
                accuracyTrain;
                %confusionMatTrain = confusionmat(attributesMatSubsetLabels, inferedLabels);
                inferedLabels = [];
                
                %Accuracy on validation data
                [accuracyValid inferedLabels classDistances] = functionGetAccuracyOnDataset(attributesMatValidation, attributesMatValidationLabels, ...
                    heldOutSeenClasses, attributes, outMLData.metricLearned);
                accuracyValid;
                %confusionMatValid = confusionmat(attributesMatValidationLabels, inferedLabels);
                
                %Accuracy on Test Data
                [accuracyTest inferedLabels classDistances] = functionGetAccuracyOnDataset(semanticEmbeddingsTest, labelsTestingData, ...
                    [inputData.defaultTestClassLabels], attributes, outMLData.metricLearned);
                %confusionMatTest = confusionmat(labelsTestingData, inferedLabels);
                inferedLabels = [];
                accuracyTest;
                
                %pause(1)
                arrayClassAccu(m, 1) = accuracyTrain;
                arrayClassAccu(m, 2) = accuracyValid;
                arrayClassAccu(m, 3) = accuracyTest;
                arrayClassAccu(m, 4) = lambda;
                arrayClassAccu(m, 5) = margin;
                arrayClassAccu(m, 6) = maxIterations;
                loss{m} = outMLData.totalLoss;
                metricLearned{m} = outMLData.metricLearned;
                %     arrayClassAccu(m, 4) = randomSampleIndices(randSmpleItr, 1);
                %     arrayClassAccu(m, 5) = randomSampleIndices(randSmpleItr, 2);
                m = m+1;
            end
        end
    end
end

close(handleW);
toc
mean(arrayClassAccu)
% distanceMahalanobis = functionGetDistancesBetweenSamples([semanticEmbeddingsTrain; semanticEmbeddingsTest] ...
%     , 'metric', metricLearned);

% distanceMahalanobis = functionGetDistancesBetweenSamples(semanticEmbeddingsTest, ...
%     attributes(:, inputData.defaultTestClassLabels)', 'metric', metricLearned);
%% END >> Metric learning



