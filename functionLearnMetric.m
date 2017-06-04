% Inputs:
% *inData: P x Q matrix with P data points each of Q dimension
% *inLabels: P x 1 matrix containing labels of P data points in inData
%inNumberOfSamplesPerClassinNumberOfClasses
% Outputs:
% *outM: Metric learned which is a matrix of size Q x Q

%%%***DistLearnKit****%%%

function outM = functionLearnMetric(inData, inLabels, inNumberOfSamplesPerClass, inNumberOfClasses)

%Initialisation
alpha = 0.95;
lambda = 0.01;
margin = 100;
dataDim = size(inData, 2);
numberOfDataSamples = size(inData, 1);
M = eye(dataDim, dataDim);
distanceMatrix = zeros(numberOfDataSamples, numberOfDataSamples);
%outerProductMatrix = zeros(numberOfDataSamples, numberOfDataSamples);
gradientGtTerm1 = zeros(dataDim, dataDim);
gradientGtTerm2 = gradientGtTerm1;
g = 1;
maxIterations = 100;
totalLoss = zeros(maxIterations + 1, 1);
totalLoss(1) = 10;
arrayLambda(1) = lambda;
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
    costMat = distanceMatrix.*blockDiagonalOnesMat + margin*blockDiagonalOnesMat - blockDiagonalOnesMat.*minValueMat;
    costMat = (costMat > 0);
    
    for sampleIndex = 1:size(distanceMatrix, 1)
        indices = find(costMat(sampleIndex, :));
        ijkTouple = [sampleIndex * ones(1, length(indices)); indices; classKIndices(sampleIndex) * ones(1, length(indices))];
        indexSetOfIJK = [indexSetOfIJK; ijkTouple'];
    end
    
    %second term of gradient
    loss2 = 0;
    gradientGtTerm2 = gradientGtTerm2 * 0;
    
    for t = 1:size(indexSetOfIJK, 1)
        i = indexSetOfIJK(t,1); j = indexSetOfIJK(t,2); k = indexSetOfIJK(t,3);
        index1 = (i - 1) * numberOfDataSamples + j;
        index2 = (i - 1) * numberOfDataSamples + k;
        outerProducts = outerProductMatrix{index1} - outerProductMatrix{index2};
        gradientGtTerm2 = gradientGtTerm2 + outerProducts;
        loss2 = loss2 + max(trace(M*outerProducts) + margin, 0);
     end
    
    gradientGt = (1 - alpha) * gradientGtTerm1 + alpha * gradientGtTerm2;
    M = M - lambda * gradientGt;
    tmpM = M;
    M = (M + M')/2;
    all(eig(M) > 0);
    [eigVec eigVal] = eig(M);
    eigVal(eigVal < 0) = 0;
    M = eigVec * eigVal * eigVec';
    all(eig(M) > 0);
    totalLoss(iteration + 1) = (1 - alpha) * loss1 + alpha * loss2; 
%     if totalLoss(iteration + 1) - totalLoss(iteration) < 0
%         lambda = 1.1*lambda;
%     else
%         lambda = 0.5*lambda;
%     end
%         arrayLambda(iteration + 1) = lambda;

   
end
toc
figure; plot(totalLoss(2:end));title('Loss')
% figure;plot(arrayLambda);title('Lambda');
outM = M;










