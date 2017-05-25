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
lambda = 0.01;
dataDim = size(inData, 2);
numberOfDataSamples = size(inData, 1);
M = eye(dataDim, dataDim);
distanceMatrix = zeros(numberOfDataSamples, numberOfDataSamples);
%outerProductMatrix = zeros(numberOfDataSamples, numberOfDataSamples);
gradientGtTerm1 = zeros(dataDim, dataDim);
gradientGtTerm2 = gradientGtTerm1;
g = 1;
maxIterations = 20;

%Gradient first term
loss1 = 0;

for classI = 1: inNumberOfClasses
    indexI = (classI - 1)*inNumberOfSamplesPerClass + 1;
    indexJ = indexI + inNumberOfSamplesPerClass - 1;
    for k = indexI:indexJ - 1
        for p = k + 1: indexJ
            Xij = (inData(k, :) - inData(p, :))' * (inData(k, :) - inData(p, :));
            gradientGtTerm1 = gradientGtTerm1 + Xij;
            loss1 = loss1 + trace(M*Xij);
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

tempAllOneMat = ones(size(distanceMatrix));
blockMat = ones(inNumberOfSamplesPerClass, inNumberOfSamplesPerClass);

for p = 1:inNumberOfClasses
    matcell{p} = blockMat;
end

blockDiagonalOnesMat = blkdiag(matcell{:});
temp2Mat = tempAllOneMat - blockDiagonalOnesMat;
totalLoss = zeros(maxIterations, 1);
tic

%Gradient descend
for iteration = 1:maxIterations
    iteration
    %FInd the distance matrix
    k = 1;
    %TODO: Only find upper/lower diagonal elements
    for row = 1:numberOfDataSamples
        for col = 1:numberOfDataSamples
            distanceMatrix(row, col) = trace(M * cell2mat(outerProductMatrix(k)));
            k = k + 1;
        end
    end
    
    betweenClassDistanceMat = distanceMatrix.*temp2Mat;
    blockDiagonalOnesMat = 10^5*blockDiagonalOnesMat;
    betweenClassDistanceMat = betweenClassDistanceMat + blockDiagonalOnesMat;
    blockDiagonalOnesMat = blockDiagonalOnesMat / 10^5;
    indexSetOfIJK = [];
    
    [minValue classKIndices] = min(betweenClassDistanceMat);
    minValueMat = repmat(minValue, length(minValue), 1);
    minValueMat = minValueMat';
    costMat = distanceMatrix.*blockDiagonalOnesMat + 1.*blockDiagonalOnesMat - blockDiagonalOnesMat.*minValueMat;
    costMat = (costMat > 0);
    
    for sampleIndex = 1:size(distanceMatrix, 1)
        indices = find(costMat(sampleIndex, :));
        ijkTouple = [sampleIndex * ones(1, length(indices)); indices; classKIndices(sampleIndex) * ones(1, length(indices))];
        indexSetOfIJK = [indexSetOfIJK; ijkTouple'];
    end
    
    %second term of gradient
    loss2 = 0;
    for t = 1:length(indexSetOfIJK)
        i = indexSetOfIJK(t,1); j = indexSetOfIJK(t,2); k = indexSetOfIJK(t,3);
        index1 = (i - 1) * numberOfDataSamples + j;
        index2 = (i - 1) * numberOfDataSamples + k;
        outerProducts = outerProductMatrix{index1} - outerProductMatrix{index2};
        gradientGtTerm2 = gradientGtTerm2 + outerProducts;
        loss2 = loss2 + trace(M*outerProducts);
    end
    
    gradientGt = (1 - alpha) * gradientGtTerm1 + alpha * gradientGtTerm2;
    M = M - lambda * gradientGt;
    [eigVec eigVal] = eig(M);
    eigVec = real(eigVec);
    eigVal = real(eigVal);
    eigVal(eigVal < 0) = 0;
    M = eigVec * eigVal * eigVec';
    all(eig(M) > 0)
    totalLoss(iteration) = (1 - alpha) * loss1 + alpha * loss2; 
%     increamentalGain = 100*(totalLoss(iteration + 1) - totalLoss(iteration))/totalLoss(iteration + 1);
%     if increamentalGain < 1
%         break;
%     end    
end
toc
k = 1;
figure; plot(totalLoss);
outM = M;










