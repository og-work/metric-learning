function outInputData = functionLoadInputs(inDATASET, inBASE_PATH, inSemanticSpace)

if(strcmp(inDATASET, 'AwA'))
    dataset_path = sprintf('%s/data/code-data/semantic-similarity/precomputed-features-AwA/', inBASE_PATH);
    %'data/code-data/semantic-similarity/cnn-features
    %temp = load(sprintf('%s/data/code-data/semantic-similarity/cnn-features/AwA/feat-imagenet-vgg-verydeep-19.mat', BASE_PATH));
    temp = load(sprintf('%s/AwA_All_vgg19Features.mat', dataset_path));
    outInputData.vggFeatures = temp.vggFeatures';
    clear temp;
    
    if strcmp(inSemanticSpace, 'word2vec')
        attributes = load(sprintf('%s/AwA_All_ClassLabelPhraseDict.mat', dataset_path));
        outInputData.attributes = attributes.phrasevec_mat';
    else
        attributes = load(sprintf('%s//data/dataset/AwA/Animals_with_Attributes/predicate-matrix-binary.txt', inBASE_PATH));
        outInputData.attributes = attributes';
    end
    temp = load(sprintf('%s/AwA_All_DatasetLabels.mat', dataset_path));
    clear tmp;
    outInputData.datasetLabels = temp.datasetLabels';
    outInputData.NUMBER_OF_CLASSES = 50;
    outInputData.defaultTestClassLabels = [8 17 22 24 26 33 34 37 38 40];
    outInputData.numberOfSamplesPerTrainClass = 92;%92;%150 apy, 92 AwA
    outInputData.classNames = {'antelope';'grizzly+bear';'killer+whale';'beaver';'dalmatian';'persian+cat';'horse';'german+shepherd';'blue+whale';'siamese+cat';'skunk';'mole';'tiger';'hippopotamus';'leopard';'moose';'spider+monkey';'humpback+whale';'elephant';'gorilla';'ox';'fox';'sheep';'seal';'chimpanzee';'hamster';'squirrel';'rhinoceros';'rabbit';'bat';'giraffe';'wolf';'chihuahua';'rat';'weasel';'otter';'buffalo';'zebra';'giant+panda';'deer';'bobcat';'pig';'lion';'mouse';'polar+bear';'collie';'walrus';'raccoon';'cow';'dolphin'};
elseif (strcmp(inDATASET, 'Pascal-Yahoo'))
    dataset_path = sprintf('%s/data/code-data/semantic-similarity/cnn-features/aPY/', inBASE_PATH);
    tmp = load(sprintf('%s/cnn_feat_imagenet-vgg-verydeep-19.mat', dataset_path));
    outInputData.vggFeatures = tmp.cnn_feat;
    clear tmp;
    tmp = load(sprintf('%s/class_attributes.mat', dataset_path));
    outInputData.datasetLabels = tmp.labels;

    if strcmp(inSemanticSpace, 'attributes')        
        outInputData.attributes = tmp.class_attributes';
    else
        tmp1 = load(sprintf('%s/apy-word2vec-300d.mat', dataset_path));
        outInputData.attributes = tmp1.word;
        clear tmp1;
    end
    
    clear tmp;
    outInputData.NUMBER_OF_CLASSES = 32;
    outInputData.defaultTestClassLabels = [21:32]%[12 14 2 32 21 22 26 8 5 29];
    outInputData.numberOfSamplesPerTrainClass = 51%150; %150 apy, 92 AwA
    outInputData.classNames = {'1 aeroplane', '2 bicycle', '3 bird', '4 boat', '5 bottle', '6 bus', '7 car', '8 cat', ...
        '9 chair','10 cow','11 diningtable','12 dog','13 horse','14 motorbike','15 person','16 pottedplant','17 sheep',...
        '18 sofa','19 train','20 tvmonitor','21 donkey','22 monkey','23 goat','24 wolf','25 jetski','26 zebra','27 centaur','28 mug',...
        '29 statue','30 building','31 bag','32 carriage'}';
else
    sprintf('No Dataset selected ...')
end
outInputData.dataset_path = dataset_path;
