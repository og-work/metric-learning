function outInputData = functionLoadInputs(inDATASET, inBASE_PATH)

if(strcmp(inDATASET, 'AwA'))
    outInputData.dataset_path = sprintf('%s/data/code-data/semantic-similarity/precomputed-features-AwA/', inBASE_PATH);
    %'data/code-data/semantic-similarity/cnn-features
    %temp = load(sprintf('%s/data/code-data/semantic-similarity/cnn-features/AwA/feat-imagenet-vgg-verydeep-19.mat', BASE_PATH));
    temp = load(sprintf('%s/AwA_All_vgg19Features.mat', dataset_path));
    vggFeatures = temp.vggFeatures;
    attributes = load(sprintf('%s/AwA_All_ClassLabelPhraseDict.mat', dataset_path));
    temp = load(sprintf('%s/AwA_All_DatasetLabels.mat', dataset_path));
    outInputData.datasetLabels = temp.datasetLabels';
    outInputData.vggFeatures = vggFeatures';
    outInputData.attributes = attributes.phrasevec_mat';
    outInputData.NUMBER_OF_CLASSES = 50;
    %Default setting of AwA
    %defaultTrainClassLabels = [1:7, 9:16, 18:21, 23, 25, 27:32, 35:36, 39, 41:50];
    % From dataset
    outInputData.defaultTestClassLabels = [8 17 22 24 26 33 34 37 38 40];
    % from semsnticdemo
    %defaultTestClassLabels = [25 39 15 6 42 14 18 48 34 24];
    outInputData.numberOfSamplesPerTrainClass = 20;%92;%150 apy, 92 AwA
elseif (strcmp(inDATASET, 'Pascal-Yahoo'))
    dataset_path = sprintf('%s/data/code-data/semantic-similarity/cnn-features/aPY/', inBASE_PATH);
    load(sprintf('%s/class_attributes.mat', dataset_path));
    load(sprintf('%s/cnn_feat_imagenet-vgg-verydeep-19.mat', dataset_path));
    outInputData.datasetLabels = labels;
    clear labels;
    outInputData.vggFeatures = cnn_feat;
    outInputData.attributes = class_attributes';
    outInputData.NUMBER_OF_CLASSES = 32;
    % From dataset
    %defaultTestClassLabels = [1 2 5 6 21 8 10 12 32];%[21:32];
    outInputData.defaultTestClassLabels = [21:32];
    outInputData.numberOfSamplesPerTrainClass = 150; %150 apy, 92 AwA
    outInputData.classNames = {'1 aeroplane', '2 bicycle', '3 bird', '4 boat', '5 bottle', '6 bus', '7 car', '8 cat', ...
        '9 chair','10 cow','11 diningtable','12 dog','13 horse','14 motorbike','15 person','16 pottedplant','17 sheep',...
        '18 sofa','19 train','20 tvmonitor','21 donkey','22 monkey','23 goat','24 wolf','25 jetski','26 zebra','27 centaur','28 mug',...
        '29 statue','30 building','31 bag','32 carriage'}';
    %defaultTestClassLabels = [1 2 5 6 21 8 10 12 32];%[21:32]; 
else
    sprintf('No Dataset selected ...')
end
    outInputData.dataset_path = dataset_path;
