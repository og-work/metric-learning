function outData = functionGenerated2DData(varargin)

inData = varargin{1};
meanVec = zeros(inData.dataDimension, inData.numberOfClasses);
sigmaVec = zeros(inData.dataDimension, inData.numberOfClasses);

for p = 1:inData.numberOfClasses
    meanVec(:, p) = randperm(100, inData.dataDimension)';
    sigmaVec(:, p) = randperm(5, inData.dataDimension)';
end

%rng(1); % For reproducibility
X = [];
% colorMap = hsv(inData.numberOfClasses);
% color = [];
labels = [];
importData = 0;
if importData
    a1 = load('test1.out');
    a2 = load('test2.out');
    a3 = load('test3.out');
    a4 = load('test4.out');
    a5 = load('test5.out');
    a6 = load('test6.out');    
    a7 = load('test7.out');
    a8 = load('test8.out');
    a9 = load('test9.out');
    a10 = load('test10.out');
    X = [[a1 a2];[a3 a4];[a5 a6];[a7 a8];[a9 a10]];
end

for p = 1:inData.numberOfClasses
    if ~importData
        X = [X; mvnrnd(meanVec(:, p)', sigmaVec(:, p)', inData.numberOfSamplesPerClass)];
    end
    %     color = [color; repmat(colorMap(p, :), inData.numberOfSamplesPerClass, 1)];
    labels = [labels; ones(inData.numberOfSamplesPerClass, 1) * p];
end

% sz = 50;
% scatter(X(:,1),X(:,2),sz, color, 'filled')
% hold on
outData.data = X;
outData.meanVec = meanVec;
outData.sigmaVec = sigmaVec;
outData.labels = labels;