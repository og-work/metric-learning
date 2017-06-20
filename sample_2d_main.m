% Sample program to check top push metric learning on 2D data

% close all
clear;
%% Generate synthetic data
inputDataGen.numberOfClasses = 12;
inputDataGen.dataDimension = 2;
inputDataGen.numberOfSamplesPerClass = 5;
data2D = functionGenerated2DData(inputDataGen);

%% Plot data
[maxV maxI] = max(data2D.data);
[minV minI] = min(data2D.data);
minMat = repmat(minV, size(data2D.data, 1), 1);
maxMat = repmat(maxV, size(data2D.data, 1), 1);
normalisedData = (data2D.data - minMat)./(maxMat - minMat);
plotData.data = normalisedData;
plotData.labels = data2D.labels;
plotData.numberOfSamplesPerClass = inputDataGen.numberOfSamplesPerClass;
plotData.numberOfClasses = inputDataGen.numberOfClasses;
funtionMySCATTERPLOT(plotData);title('Normalised input data');
plotData.data = data2D.data;
funtionMySCATTERPLOT(plotData);[h, ~] = legend('show');title('Un-normalised input data');
plotData.data = [];

%% Prepare data for ML
lambdaArray = 0.01:0.05:2;
marginArray = 1:100:1000;
maxIterationsArray = 100:100:500;
inDataML.data = data2D.data;
inDataML.labels = data2D.labels;
inDataML.numberOfSamplesPerClass = inputDataGen.numberOfSamplesPerClass;
inDataML.numberOfClasses = inputDataGen.numberOfClasses;

%Metric learning
for lambda = lambdaArray
    for margin = marginArray
        for maxIterations = maxIterationsArray
            inDataML.lambda = 0.001;%lambda;
            inDataML.margin = 100;%margin;
            inDataML.maxIterations = 100;%maxIterations;
            inDataML.normalisedData = normalisedData; % Only for debug
            inDataML.trainClasses = 1:inputDataGen.numberOfClasses;
            outputDataML = functionLearnMetric(inDataML);            
                  
            for m = 1:inDataML.maxIterations
                % Transform data from learned metric
                metricLearned = outputDataML.arrayM{m};
                dataTransformed = functionTransformPoints(data2D.data, data2D.labels, metricLearned);
                [maxV maxI] = max(dataTransformed.data);
                [minV minI] = min(dataTransformed.data);
                minMat = repmat(minV, size(dataTransformed.data, 1), 1);
                maxMat = repmat(maxV, size(dataTransformed.data, 1), 1);
                normalisedTransformedData = (dataTransformed.data - minMat)./(maxMat - minMat);
                plotData.data = normalisedTransformedData;
                % Plot transformed data
                funtionMySCATTERPLOT(plotData);title('Normalised ***output*** data');
                plotData.data = dataTransformed.data;
                %funtionMySCATTERPLOT(plotData);title('Un-Normalised ***output*** data');
                saveas(gcf, sprintf('data/plots/plot-%02d.png', m));
            end
        end
    end
end

imageNames = dir(fullfile('data/plots','*.png'));
imageNames = {imageNames.name}';
outputVideo = VideoWriter(fullfile('data/plots/','out2.avi'));
outputVideo.FrameRate = 1;
open(outputVideo)
for ii = 1:length(imageNames)
   img = imread(fullfile('data/plots', imageNames{ii}));
   writeVideo(outputVideo,img)
end
close(outputVideo)
close all
save('data/var.mat', 'outputDataML');
save('data/allVar.mat')
close all
