
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
VIEW_TSNE = 0;

%% START >> Load data

inputData = functionLoadInputs(DATASET, BASE_PATH);
%Normalise attributes feature
% inputData.attributes = inputData.attributes';
% inputData.attributes  = inputData.attributes./repmat...
%     (sum(inputData.attributes ,2),1,size(inputData.attributes ,2));
% inputData.attributes = inputData.attributes';

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
vggFeaturesTesting = [];
labelsTrainingData = [];
labelsTestingData = [];
indicesOfTrainingSamples = [];
indicesOfTestingSamples = [];

for classInd = 1:length(defaultTrainClassLabels)
    tmp = [];
    tmp = (inputData.datasetLabels == defaultTrainClassLabels(classInd));
    indicesOfTrainingSamples = [indicesOfTrainingSamples; find(tmp)];
    vggFeaturesTraining = [vggFeaturesTraining inputData.vggFeatures(:, find(tmp))];
    labelsTrainingData = [labelsTrainingData; defaultTrainClassLabels(classInd) * ones(sum(tmp), 1)];
end


for classInd = 1:length(inputData.defaultTestClassLabels)
    tmp = [];
    tmp = (inputData.datasetLabels == inputData.defaultTestClassLabels(classInd));
    indicesOfTestingSamples = [indicesOfTestingSamples; find(tmp)];
    vggFeaturesTesting = [vggFeaturesTesting, inputData.vggFeatures(:, find(tmp))];
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

if VIEW_TSNE
    
    funtionVisualiseData(inputData.vggFeatures(:, tempC)', ...
        labelsTrainingSubsetData, inputData.classNames, inputData.NUMBER_OF_CLASSES, ...
        'VGG features training');
    
    %Plot seen and unseen samples
    funtionVisualiseData([semanticEmbeddingsTrain; semanticEmbeddingsTest], ...
        [labelsTrainingSubsetData; labelsTestingData], inputData.classNames, inputData.NUMBER_OF_CLASSES, ...
        'Seen and Unseen class samples MAPPED');
    
    % %Prototypes
    funtionVisualiseData(inputData.attributes', ...
        [1:inputData.NUMBER_OF_CLASSES], inputData.classNames, inputData.NUMBER_OF_CLASSES, ...
        'Attributes (prototypes) seen and unseen classes');
    
    %plot seen points
    funtionVisualiseData(semanticEmbeddingsTrain, ...
        labelsTrainingSubsetData, inputData.classNames, length(trainClassNames), ...
        'Seen class samples MAPPED');
    
    %Plot unseen class points with unseen prototypes
    unseenPrototypes = inputData.attributes(:, inputData.defaultTestClassLabels);
    unseenPrototypesLabels = [inputData.NUMBER_OF_CLASSES + 1 : inputData.NUMBER_OF_CLASSES + length(inputData.defaultTestClassLabels)];
    ext = {'21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32'};
    classNamesExtended = [inputData.classNames; ext'];
    funtionVisualiseData([semanticEmbeddingsTest; unseenPrototypes'], ...
        [labelsTestingData; unseenPrototypesLabels'], classNamesExtended, 2*length(testClassNames), ...
        'Embedded Unseen class samples and unseen prototypes');
end


%% Start >> Metric learning
%Prepare data for stochastic gradient descend
m = 1;
for class1 = [1:18]
    for class2 = [class1+1: 19]
        for class3 = [class2 + 1: 20]
        [class1 class2 class3]
        numberOfSamplesForSGDPerClass = 30;
        attributesMatSubset = [];%zeros(numberOfSamplesForSGDPerClass * length(trainClassNames), size(attributesMat, 2));
        attributesMatSubsetLabels = [];
        tmpTrainClasses = [class1, class2, class3];
        
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
        % accuracyTest = functionGetAccuracyOnDataset(semanticEmbeddingsTest, labelsTestingData, ...
        %     inputData.defaultTestClassLabels, inputData.attributes, metricLearned)
        
        %Accuracy on train data
        % accuracyTrain = functionGetAccuracyOnDataset(attributesMatSubset, attributesMatSubsetLabels, ...
        %     defaultTrainClassLabels, inputData.attributes, metricLearned)
        [accuracyTrain inferedLabels classDistances] = functionGetAccuracyOnDataset(attributesMatSubset, attributesMatSubsetLabels, ...
            tmpTrainClasses, inputData.attributes, metricLearned);
        accuracyTrain;
        inferedLabels';
        
        [accuracyValid inferedLabels classDistances] = functionGetAccuracyOnDataset(attributesMatValidation, attributesMatValidationLabels, ...
            tmpTrainClasses, inputData.attributes, metricLearned);
        accuracyValid;
        
        arrayClassAccu(m, 1) = accuracyTrain;
        arrayClassAccu(m, 2) = accuracyValid;
        arrayClassAccu(m, 3) = class1;
        arrayClassAccu(m, 4) = class2;
        arrayClassAccu(m, 5) = class3;
        m = m+1;
        end
    end
end

% End >> Testing
%% END >> Metric learning



