%This function calculates the kernel from data. 
% *Inputs:
% BASE_PATH
% Base path for directory based on system/platform
% 
% inData:
% matrix NxD of features  where N: number of points and D: feature dimension
% i.e. Points are arranged along the rows
% 
% inDataset: Name of dataset
% *Outputs:
% kernel:
% Kernel computed from the data

function outKernel = functionGetKernel(inBASE_PATH, inData, inKernelType, inDatasetPath, inDataset, inNormFlag)

% Precompute Kernel Matrix
addpath(genpath(sprintf('%s/codes/matlab-stuff/tree-based-zsl', inBASE_PATH)));

if exist(fullfile(inDatasetPath, sprintf('%s_%s_kernel_full_dataset_vgg.mat', inDataset, inKernelType)),'file')
    fprintf('Loading the precomputed kernel...%s \n', inKernelType)
    temp = load(strcat(inDatasetPath, sprintf('%s_%s_kernel_full_dataset_vgg.mat', inDataset, inKernelType)));
    outKernel = temp.outKernel;
else
    features = [];
    if inNormFlag
        %%% Normalize Feature Vector and Label Vector
        temp.FeatureMatrix = func_NormalizeFeatureMatrix(inData);
    else
        temp.FeatureMatrix = inData;
    end
    
    features = [features temp.FeatureMatrix];
    features(isnan(features)) = 0;
    
    %It may take long time to compute Chi2 kernel
    fprintf('Computing the kernel...%s. It may take a long while.\n', inKernelType)
    outKernel = func_PrecomputeKernel(features, features, inKernelType);
    save(sprintf('%s/%s_%s_kernel_full_dataset_vgg.mat', inDatasetPath, inDataset, inKernelType), 'outKernel', '-v7.3');

end