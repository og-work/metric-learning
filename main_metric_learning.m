
clc;
clear
close all;

% 1: Linux Laptop
% 2: Windows laptop
% 3: Linux Desktop
% 4: Windows Desktop
SYSTEM_PLATFORM = 3;
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

%% Load data
inputData = functionLoadInputs(DATASET, BASE_PATH, semanticSpace);

%% Normalise attributes feature
attributes = functionNormaliseData(inputData.attributes', normType);
vggFeatures =  functionNormaliseData(inputData.vggFeatures', normType);

%% Get train and test classes labels/indices
labels = zeros(1, inputData.NUMBER_OF_CLASSES);
labels(inputData.defaultTestClassLabels) = 1;
labels = 1. - labels;
defaultTrainClassLabels = find(labels);
trainClassNames = inputData.classNames(defaultTrainClassLabels);
testClassNames = inputData.classNames(inputData.defaultTestClassLabels);

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

%% Get kernelised data
if useKernelisedData
    normalise = 0;
    kernelFullData = functionGetKernel(BASE_PATH, vggFeatures', kernelType, inputData.dataset_path, ...
        DATASET, normalise);
    kernelTrainData = kernelFullData(indicesOfTrainingSamples, indicesOfTrainingSamples);
    kernelTestData = kernelFullData(indicesOfTestingSamples, indicesOfTrainingSamples);
end

%% Get semantic vectors per train samples
attributesMat = [];
labelsTrainingSubsetData = [];
tempC = [];

%Prepare semantic features data by selecting fixed number of samples per class
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

%% Train regressor
if 1
    [semanticEmbeddingsTrain mappingF semanticEmbeddingsTest]= functionTrainRegressor(regressorInputData', ...
        attributesMat, BASE_PATH, useKernelisedData, indicesOfTrainingSamplesSubset, indicesOfTestingSamples);
    regressorData.semanticEmbeddingsTrain = semanticEmbeddingsTrain;
    regressorData.semanticEmbeddingsTest = semanticEmbeddingsTest;
    save(sprintf('data/%s-regressor-vgg-%s-%s.mat', DATASET, semanticSpace, kernelType), 'regressorData');
else
    regD = load(sprintf('data/%s-regressor-vgg-%s-%s.mat', DATASET, semanticSpace, kernelType));
    semanticEmbeddingsTrain = regD.regressorData.semanticEmbeddingsTrain;
    semanticEmbeddingsTest = regD.regressorData.semanticEmbeddingsTest;
end
% semanticEmbeddingsTest(semanticEmbeddingsTest < 0) = 0;
% semanticEmbeddingsTrain(semanticEmbeddingsTrain < 0) = 0;

figure;
distanceMatrix = funtionPlotConfusionMatrixGetDistanceMatrix(semanticEmbeddingsTrain, attributes(:, defaultTrainClassLabels)', ...
    labelsTrainingSubsetData, 'l2', [], defaultTrainClassLabels);
title('Training data');

figure;
[distanceMatrixTest perConfusionMatTest]= funtionPlotConfusionMatrixGetDistanceMatrix(semanticEmbeddingsTest, attributes(:, inputData.defaultTestClassLabels)', ...
    labelsTestingData, 'l2', [], inputData.defaultTestClassLabels);
title('Without metric learning');

%% Accuracy on Test Data without ML
[accuracyTestWithoutML inferedLabels classDistances] = functionGetAccuracyOnDataset(semanticEmbeddingsTest, labelsTestingData, ...
    [inputData.defaultTestClassLabels], attributes, eye(size(semanticEmbeddingsTest, 2)));
% = confusionmat(labelsTestingData, inferedLabels);
accuracyTestWithoutML
inferedLabels = [];classDistances = [];
% distanceMatrix = pdist2(semanticEmbeddingsTrain, semanticEmbeddingsTrain, 'euclidean');

%% tSNE visualisation
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

%% Start >> Metric learning using held out classes
numberOfNonHeldOutClasses = ceil(0.8*length(defaultTrainClassLabels));
indices = randperm(length(defaultTrainClassLabels), numberOfNonHeldOutClasses);
trainClassesNonHeldOut = defaultTrainClassLabels(indices);
tmplabels = zeros(1, (inputData.NUMBER_OF_CLASSES));
tmplabels(trainClassesNonHeldOut) = 1;
tmplabels(inputData.defaultTestClassLabels) = 1;
tmplabels = 1. - tmplabels;
trainClassesheldOut = find(tmplabels);

%Validation Data for metric learning
%numberOfPerClassSamplesForValidation = inputData.numberOfSamplesPerTrainClass - numberOfSamplesForSGDPerClass;
numberOfPerClassSamplesForValidation = inputData.numberOfSamplesPerTrainClass;
attributesMatValidation = [];
attributesMatValidationLabels = [];

for p = trainClassesheldOut%length(defaultTrainClassLabels)%length(trainClassNames)
    %     class = find(defaultTrainClassLabels == p);
    %     startI = inputData.numberOfSamplesPerTrainClass * (class - 1) + 1;
    %     endI = startI + numberOfPerClassSamplesForValidation - 1;
    %     attributesMatValidation = [attributesMatValidation; semanticEmbeddingsTrain(startI:endI, :)];
    %     attributesMatValidationLabels = [attributesMatValidationLabels; labelsTrainingSubsetData(startI:endI)];
    tp1 = find(labelsTrainingSubsetData == p);
    tp2 = tp1(1:numberOfPerClassSamplesForValidation);
    attributesMatValidation = [attributesMatValidation; semanticEmbeddingsTrain(tp2, :)];
    attributesMatValidationLabels = [attributesMatValidationLabels; labelsTrainingSubsetData(tp2)];
end

%Prepare data for stochastic gradient descend
numberOfSamplesForSGDPerClass = 5;
randomItrMax = 1;
randomSampleIndices = [];

% Generate tuples of random samples
for b = 1:randomItrMax
    %randomSampleIndices(:, b) = randi([1 inputData.numberOfSamplesPerTrainClass], randomItrMax, 1);
    randomSampleIndices(b, :)  = randperm(inputData.numberOfSamplesPerTrainClass, numberOfSamplesForSGDPerClass);
end

tic
% handleW = waitbar(0,'Random sampling in progress...');
arrayClassAccu = [];
tp1= []; tp2 = [];

for randSmpleItr = 1:randomItrMax
    attributesMatSubset = [];%zeros(numberOfSamplesForSGDPerClass * length(trainClassNames), size(attributesMat, 2));
    attributesMatSubsetLabels = [];
    
    %Train data for metric learning
    for p = trainClassesNonHeldOut
        %         classIn = find(defaultTrainClassLabels == p);
        %         startI = inputData.numberOfSamplesPerTrainClass * (classIn - 1);
        %         sampleIndices = startI + randomSampleIndices(randSmpleItr, :);
        %         attributesMatSubset = [attributesMatSubset; semanticEmbeddingsTrain(sampleIndices, :)];
        %         attributesMatSubsetLabels = [attributesMatSubsetLabels; labelsTrainingSubsetData(sampleIndices)];
        tp1 = find(labelsTrainingSubsetData == p);
        tp2 = tp1(randomSampleIndices(randSmpleItr, :));
        attributesMatSubset = [attributesMatSubset; semanticEmbeddingsTrain(tp2, :)];
        attributesMatSubsetLabels = [attributesMatSubsetLabels; labelsTrainingSubsetData(tp2)];
    end
    
    
    %Use only seen class prototypes for metric learning
    %     attributesMatSubset = attributes(:, trainClassesNonHeldOut)';
    %     attributesMatSubsetLabels = trainClassesNonHeldOut';%defaultTrainClassLabels';
    
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
    inputDataMetricLearn.numberOfSamplesForSGDPerClass = numberOfSamplesForSGDPerClass;
    inputDataMetricLearn.numberOfClasses = length(trainClassesNonHeldOut);
    inputDataMetricLearn.trainClasses = trainClassesNonHeldOut;%defaultTrainClassLabels;
    lambdaArray = 0.001:0.2:2;
    marginArray = 50:30:1000;
    maxIterationsArray = 150;
    maxCount = length(lambdaArray)*length(marginArray)*length(maxIterationsArray);
    loss = {};
    metricLearned = {};
    m = 1;
    
    for lambda = lambdaArray
        for margin = 10%marginArray
            for maxIterations = maxIterationsArray
                %                 waitbar(m/maxCount);
                inputDataMetricLearn.lambda = lambda;
                inputDataMetricLearn.margin = margin;
                inputDataMetricLearn.maxIterations = maxIterations;
                inputDataMetricLearn.normalisedData = [];
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
                    trainClassesNonHeldOut, attributes, outMLData.metricLearned);
                %confusionMatTrain = confusionmat(attributesMatSubsetLabels, inferedLabels);
%                 figure;
%                 [distanceMatrixTrain perConfusionMatTrain]= funtionPlotConfusionMatrixGetDistanceMatrix(attributesMatSubset, attributes(:, trainClassesNonHeldOut)', ...
%                     attributesMatSubsetLabels, 'metric', outMLData.metricLearned, trainClassesNonHeldOut);
%                 title('training data metric learning');
                inferedLabels = [];
                
                %Accuracy on validation data
                [accuracyValid inferedLabels classDistances] = functionGetAccuracyOnDataset(attributesMatValidation, attributesMatValidationLabels, ...
                    trainClassesheldOut, attributes, outMLData.metricLearned);
                confusionMatValid = confusionmat(attributesMatValidationLabels, inferedLabels);
%                 figure;
%                 [distanceMatrixValid perConfusionMatValid]= funtionPlotConfusionMatrixGetDistanceMatrix(attributesMatValidation, attributes(:, trainClassesheldOut)', ...
%                     attributesMatValidationLabels, 'metric', outMLData.metricLearned, trainClassesheldOut);
%                 title('Validation data metric learning');
                
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

% close(handleW);
toc
mean(arrayClassAccu)

%% Test
[val ind] = max(arrayClassAccu(:, 2));
inputDataMetricLearn.lambda = arrayClassAccu(ind, 4);
inputDataMetricLearn.margin = arrayClassAccu(ind, 5);
inputDataMetricLearn.maxIterations = arrayClassAccu(ind, 6);
inputDataMetricLearn.numberOfSamplesPerClass = numberOfSamplesForSGDPerClass;
inputDataMetricLearn.numberOfClasses = length(defaultTrainClassLabels);
inputDataMetricLearn.trainClasses = defaultTrainClassLabels;


%Prepare data for metric learning
attributesMatSubset = [];
attributesMatSubsetLabels = [];

for p = defaultTrainClassLabels
    classIn = find(defaultTrainClassLabels == p);
    startI = inputData.numberOfSamplesPerTrainClass * (classIn - 1);
    sampleIndices = startI + [1:inputDataMetricLearn.numberOfSamplesPerClass];
    attributesMatSubset = [attributesMatSubset; semanticEmbeddingsTrain(sampleIndices, :)];
    attributesMatSubsetLabels = [attributesMatSubsetLabels; labelsTrainingSubsetData(sampleIndices)];
end

inputDataMetricLearn.data = attributesMatSubset;
inputDataMetricLearn.labels = attributesMatSubsetLabels;
outMLData = functionLearnMetric(inputDataMetricLearn);

%Accuracy on Test Data
[accuracyTestFinal inferedLabels classDistances] = functionGetAccuracyOnDataset(semanticEmbeddingsTest, labelsTestingData, ...
    [inputData.defaultTestClassLabels], attributes, outMLData.metricLearned);
%confusionMatTest = confusionmat(labelsTestingData, inferedLabels);
figure;
[distanceMatrixTest perConfusionMatTest]= funtionPlotConfusionMatrixGetDistanceMatrix(semanticEmbeddingsTest, attributes(:, inputData.defaultTestClassLabels)', ...
    labelsTestingData, 'metric', outMLData.metricLearned, inputData.defaultTestClassLabels);
title('training data metric learning');
inferedLabels = [];
accuracyTestFinal;
