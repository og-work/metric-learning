
clc;
clear
% close all;

% 1: Linux Laptop
% 2: Windows laptop
% 3: Linux Desktop
% 4: Windows Desktop
SYSTEM_PLATFORM = 1;
BASE_PATH = '';
listDatasets = {'AwA', 'Pascal-Yahoo'};
DATASET_ID = 1;
DATASET = listDatasets{DATASET_ID};
listOfKernelTypes = {'chisq', 'cosine', 'linear', 'rbf', 'rbfchisq'};
KERNEL_ID = 3;
kernelType = listOfKernelTypes{KERNEL_ID};
useKernelisedData = 1;
listNormalisationTypes = {'none', 'l1', 'l2', 'metric'};
normTypeIndex = 1;
normType = listNormalisationTypes{normTypeIndex};
%Enable/add required tool boxes
addPath = 1;
BASE_PATH = functionEnvSetup(SYSTEM_PLATFORM, addPath);
VIEW_TSNE = 0;

%% START >> Load data
inputData = functionLoadInputs(DATASET, BASE_PATH);
inputData.numberOfSamplesPerTrainClass = 92;

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
    %tr_sample_ind = tr_sample_ind + tr_sample_class_ind;
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

%Train regressor
[semanticEmbeddingsTrain mappingF semanticEmbeddingsTest]= functionTrainRegressor(regressorInputData', ...
    attributesMat, BASE_PATH, useKernelisedData, indicesOfTrainingSamplesSubset, indicesOfTestingSamples);

% semanticEmbeddingsTest(semanticEmbeddingsTest < 0) = 0;
% semanticEmbeddingsTrain(semanticEmbeddingsTrain < 0) = 0;

%% END >> Mapping of attributes

% distanceMatrix = functionGetDistancesBetweenSamples(semanticEmbeddingsTrain, 'l2');
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

m = 1;
% for class1 = [1:18]
%     for class2 = [class1+1: 19]
%         for class3 = [class2 + 1: 20]

%Prepare data for stochastic gradient descend
numberOfSamplesForSGDPerClass = 2
attributesMatSubset = [];%zeros(numberOfSamplesForSGDPerClass * length(trainClassNames), size(attributesMat, 2));
attributesMatSubsetLabels = [];
tmpTrainClasses = [1:20];

%Train data for metric learning
for p = tmpTrainClasses%length(trainClassNames)
    startI = inputData.numberOfSamplesPerTrainClass * (p - 1) + 1;
    endI = startI + numberOfSamplesForSGDPerClass - 1;
    attributesMatSubset = [attributesMatSubset; semanticEmbeddingsTrain(startI:endI, :)];
    attributesMatSubsetLabels = [attributesMatSubsetLabels; labelsTrainingSubsetData(startI:endI)];
end

%Validation Data for metric learning
numberOfPerClassSamplesForValidation = inputData.numberOfSamplesPerTrainClass - numberOfSamplesForSGDPerClass;
attributesMatValidation = [];
attributesMatValidationLabels = [];

for p = tmpTrainClasses%length(trainClassNames)
    startI = inputData.numberOfSamplesPerTrainClass * (p - 1) + 1 + numberOfSamplesForSGDPerClass;
    endI = startI + numberOfPerClassSamplesForValidation - 1;
    attributesMatValidation = [attributesMatValidation; semanticEmbeddingsTrain(startI:endI, :)];
    attributesMatValidationLabels = [attributesMatValidationLabels; labelsTrainingSubsetData(startI:endI)];
end

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

metricLearned = functionLearnMetric(attributesMatSubset, attributesMatSubsetLabels, ...
    numberOfSamplesForSGDPerClass, length(tmpTrainClasses));%;length(trainClassNames));

[eigVec eigVal] = eig((metricLearned + metricLearned')/2);

if VIEW_TSNE
    %plot seen points
    funtionVisualiseData(attributesMatSubset, ...
        attributesMatSubsetLabels, inputData.classNames, length(trainClassNames), ...
        'Subset seen class samples MAPPED');
end

% Start >> Testing
%Accuracy on Test Data
accuracyTest = functionGetAccuracyOnDataset(semanticEmbeddingsTest, labelsTestingData, ...
    [inputData.defaultTestClassLabels], attributes, metricLearned)

%Accuracy on train data
[accuracyTrain inferedLabels classDistances] = functionGetAccuracyOnDataset(attributesMatSubset, attributesMatSubsetLabels, ...
    tmpTrainClasses, attributes, metricLearned);
accuracyTrain
inferedLabels';

%Accuracy on validation data
[accuracyValid inferedLabels classDistances] = functionGetAccuracyOnDataset(attributesMatValidation, attributesMatValidationLabels, ...
    tmpTrainClasses, attributes, metricLearned);
accuracyValid

% distanceMahalanobis = functionGetDistancesBetweenSamples([semanticEmbeddingsTrain; semanticEmbeddingsTest] ...
%     , 'metric', metricLearned);

% distanceMahalanobis = functionGetDistancesBetweenSamples(semanticEmbeddingsTest, ...
%     attributes(:, inputData.defaultTestClassLabels)', 'metric', metricLearned);

%         arrayClassAccu(m, 1) = accuracyTrain;
%         arrayClassAccu(m, 2) = accuracyValid;
%         arrayClassAccu(m, 3) = class1;
%         arrayClassAccu(m, 4) = class2;
%         arrayClassAccu(m, 5) = class3;
%         m = m+1;
%         end
%     end
% end

% End >> Testing
%% END >> Metric learning



