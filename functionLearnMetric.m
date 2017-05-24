% Inputs:
% *inData: P x Q matrix with P data points each of Q dimension
% *inLabels: P x 1 matrix containing labels of P data points in inData
% 
% Outputs:
% *outM: Metric learned which is a matrix of size Q x Q

%%%***DistLearnKit****%%%

function outM = functionLearnMetric(inData, inLabels, inNumberOfSamplesPerClass, inNumberOfClasses)

%Initialisation
alpha = 0.5;
dataDim = size(inData, 2);
numberOfDataSamples = size(inData, 1);
M = eye(dataDim, dataDim);
distanceMatrix = zeros(numberOfDataSamples, numberOfDataSamples);
%outerProductMatrix = zeros(numberOfDataSamples, numberOfDataSamples);
gradientGtTerm1 = zeros(dataDim, dataDim);
gradientGtTerm2 = gradientGtTerm1;
g = 1;

%Gradient first term
for classI = 1: inNumberOfClasses
    indexI = (classI - 1)*inNumberOfSamplesPerClass + 1;
    indexJ = indexI + inNumberOfSamplesPerClass - 1;
    for k = indexI:indexJ - 1
        for p = k + 1: indexJ
            gradientGtTerm1 = gradientGtTerm1 + (inData(k, :) - inData(p, :))' * (inData(k, :) - inData(p, :));
            tmpVar(g, 1) = k; tmpVar(g, 2) = p; g = g + 1;
        end
    end
end

k = 1;
%TODO: Only find upper/lower diagonal elements
for row = 1:numberOfDataSamples
    for col = 1:numberOfDataSamples
        outerProductMatrix{k} = (inData(row, :) - inData(col, :))' * (inData(row, :) - inData(col, :));
        k = k + 1;
    end
end

%FInd the distance matrix
k = 1;
%TODO: Only find upper/lower diagonal elements
for row = 1:numberOfDataSamples
    for col = 1:numberOfDataSamples
        distanceMatrix(row, col) = trace(M * cell2mat(outerProductMatrix(k)));
        k = k + 1;
    end
end

tempAllOneMat = ones(size(distanceMatrix));
blockMat = ones(inNumberOfSamplesPerClass, inNumberOfSamplesPerClass);

for p = 1:inNumberOfClasses
    matcell{p} = blockMat;
end

blockDiagonalOnesMat = blkdiag(matcell{:});
temp2Mat = tempAllOneMat - blockDiagonalOnesMat;
betweenClassDistanceMat = distanceMatrix.*temp2Mat;
blockDiagonalOnesMat = 10^5*blockDiagonalOnesMat;
betweenClassDistanceMat = betweenClassDistanceMat + blockDiagonalOnesMat;
indexSetOfIJK = [];

% for classI = 1:inNumberOfClasses
%     rowStart = (classI - 1)*inNumberOfSamplesPerClass + 1;
%     rowEnd = rowStart + inNumberOfSamplesPerClass - 1;
%     betweenClassDistanceMatForClassI = betweenClassDistanceMat(rowStart:rowEnd, :);
%     [minVal1 classesI] = min(betweenClassDistanceMatForClassI);
%     [minVal2 classK] = min(min(betweenClassDistanceMatForClassI));
%     %classK = classesI(classK);
%     classIIndex = classI;
%     classKIndex = ceil(classK/inNumberOfSamplesPerClass);
%     indexSetOfIJK = [indexSetOfIJK; [classIIndex classIIndex classKIndex]];
% end
% 
for classI = 1:inNumberOfClasses
    rowStart = (classI - 1)*inNumberOfSamplesPerClass + 1;
    rowEnd = rowStart + inNumberOfSamplesPerClass - 1;
    betweenClassDistanceMatForClassI = betweenClassDistanceMat(rowStart:rowEnd, :);
    [minVal1 classesI] = min(betweenClassDistanceMatForClassI);
    [minVal2 classK] = min(min(betweenClassDistanceMatForClassI));
    %classK = classesI(classK);
    classIIndex = classI;
    classKIndex = ceil(classK/inNumberOfSamplesPerClass);
    indexSetOfIJK = [indexSetOfIJK; [classIIndex classIIndex classKIndex]];
end


%second term of gradient
for t = 1:length(indexSetOfIJK)
    gradientGtTerm2 = gradientGtTerm2 + outerProductMatrix{k};
end

gradientGt = (1 - alpha) * gradientGtTerm1 + alpha * gradientGtTerm2;

[minVal1 classesI] = min(betweenClassDistanceMat);
[minVal2 classK] = min(min(betweenClassDistanceMat));
classI = classesI(classK);
classIIndex = ceil(classI/inNumberOfSamplesPerClass);
classKIndex = ceil(classK/inNumberOfSamplesPerClass);
blockDiagonalOnesMat = blockDiagonalOnesMat/10^5;
minVal2Mat = blockDiagonalOnesMat*minVal2;
pushCostTriggerMatrix = blockDiagonalOnesMat.*distanceMatrix - minValMat + blockDiagonalOnesMat;
pushCostTriggerMatrix = (pushCostTriggerMatrix > 0);



 
k = 1;










