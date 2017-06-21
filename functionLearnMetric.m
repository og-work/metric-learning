% Inputs:
% *inData: P x Q matrix with P data points each of Q dimension
% *inLabels: P x 1 matrix containing labels of P data points in inData
% *inNumberOfSamplesPerClassinNumberOfClasses
% Outputs:
% *outM: Metric learned which is a matrix of size Q x Q

%%%***DistLearnKit****%%%

function outputData = functionLearnMetric(fullData)

gradientGtTerm1 = zeros(size(fullData.data, 2), size(fullData.data, 2));
gradientGtTerm2 = gradientGtTerm1;
g = 1;
%totalLoss = zeros(maxIterations + 1, 1);
totalLoss = 10;
loss1 = 0;
batchSize = 4;
numberOfBatches = fullData.numberOfClasses / batchSize;
numberOfEpochs = 100;
iteration = 1;

for epoch = 1:numberOfEpochs
    shuffleIndices = randperm(fullData.numberOfClasses, fullData.numberOfClasses);
    shuffledClasses = fullData.trainClasses(shuffleIndices);
    for batchInd = 1:numberOfBatches
        currentBatch = shuffledClasses((batchInd - 1)*batchSize + 1 ...
            : (batchInd - 1)*batchSize + batchSize);
        
        inData = [];
        inLabels = [];
        inNumberOfClasses = 0;
        
        for cl = 1:batchSize
            tp1 = find(fullData.labels == currentBatch(cl));
            tp2 = tp1(1:fullData.numberOfSamplesForSGDPerClass);
            inData = [inData; fullData.data(tp2, :)];
            inLabels = [inLabels; fullData.labels(tp2)];
            
            %             thisClassInd = find(fullData.trainClasses == currentBatch(cl));
            %             startI = (thisClassInd - 1) * fullData.numberOfSamplesPerClass + 1;
            %             endI = startI + fullData.numberOfSamplesForSGDPerClass - 1;
            %             inData = [inData; fullData.data(startI:endI, :)];
            %             inLabels = [inLabels; fullData.labels(startI:endI, :)];
            %             %inLabels = [inLabels; thisClass * ones(fullData.numberOfSamplesPerClass, 1)];
            inNumberOfClasses = inNumberOfClasses + 1;
        end
        
        % inData = inputData.data;
        % inNormalisedData = inputData.normalisedData; % Only for debug
        % inLabels = inputData.labels;
        numberOfSamplesForSGDPerClass = fullData.numberOfSamplesForSGDPerClass;
        % inNumberOfClasses = inputData.numberOfClasses;
        lambda = fullData.lambda;
        margin = fullData.margin;
        maxIterations = fullData.maxIterations;
        FAST = 0;
        %Initialisation
        alpha = 0.5; %alpha = 1 removes intra class penalty term
        
        dataDim = size(inData, 2);
        numberOfDataSamples = size(inData, 1);
        M = eye(dataDim, dataDim);
        distanceMatrix = zeros(numberOfDataSamples, numberOfDataSamples);
        %outerProductMatrix = zeros(numberOfDataSamples, numberOfDataSamples);
        arrayLambda(1) = lambda;
        
        k = 1;
        %TODO: Only find upper/lower diagonal elements
        if FAST
            for row = 1:numberOfDataSamples
                for col = row:numberOfDataSamples
                    outerProductMatrix{k} = (inData(row, :) - inData(col, :))' * (inData(row, :) - inData(col, :));
                    k = k + 1;
                end
            end
        else
            for row = 1:numberOfDataSamples
                for col = 1:numberOfDataSamples
                    outerProductMatrix{k} = (inData(row, :) - inData(col, :))' * (inData(row, :) - inData(col, :));
                    indicesOuterProduct(k, :) = [row col k];
                    k = k + 1;
                end
            end
        end
        
        tempAllOneMat = ones(size(distanceMatrix));
        blockMat = ones(numberOfSamplesForSGDPerClass, numberOfSamplesForSGDPerClass);
        
        for p = 1:inNumberOfClasses
            matcell{p} = blockMat;
        end
        
        blockDiagonalOnesMat = blkdiag(matcell{:});
        temp2Mat = tempAllOneMat - blockDiagonalOnesMat;
        tic
        
        if (rem(iteration, 10) == 0)
            iteration
        end
        
        %Gradient term 1 and loss term1
        gradientGtTerm1 = gradientGtTerm1 * 0;
        loss1 = 0;
        for classI = 1: inNumberOfClasses
            indexI = (classI - 1)*numberOfSamplesForSGDPerClass + 1;
            indexJ = indexI + numberOfSamplesForSGDPerClass - 1;
            for k = indexI:indexJ - 1
                for p = k + 1: indexJ
                    Xij = (inData(k, :) - inData(p, :))' * (inData(k, :) - inData(p, :));
                    gradientGtTerm1 = gradientGtTerm1 + Xij;
                    if  trace(M*Xij) < 0
                        error('Something wrong...distance cant be negative');
                    end
                    loss1 = loss1 + trace(M*Xij);
                    tmpVar(g, 1) = k; tmpVar(g, 2) = p; g = g + 1;
                end
            end
        end
        
        costM = [];
        distanceMatrix = 0* distanceMatrix;
        %Find the distance matrix
        k = 1;
        %TODO: Only find upper/lower diagonal elements
        if FAST
            for row = 1:numberOfDataSamples
                for col = row:numberOfDataSamples
                    distanceMatrix(row, col) = trace(M * cell2mat(outerProductMatrix(k)));
                    k = k + 1;
                end
            end
        else
            for row = 1:numberOfDataSamples
                for col = 1:numberOfDataSamples
                    distanceMatrix(row, col) = trace(M * cell2mat(outerProductMatrix(k)));
                    k = k + 1;
                end
            end
        end
        
        if FAST
            distanceMatrix = distanceMatrix + distanceMatrix';
        end
        
        maxVal = max(max(distanceMatrix));
        betweenClassDistanceMat = distanceMatrix.*temp2Mat;
        blockDiagonalOnesMat = (ceil(maxVal + 1))*blockDiagonalOnesMat;
        betweenClassDistanceMat = betweenClassDistanceMat + blockDiagonalOnesMat;
        blockDiagonalOnesMat = blockDiagonalOnesMat / (ceil(maxVal + 1));
        
        [minValue classKIndices] = min(betweenClassDistanceMat');
        minValueMat = repmat(minValue, length(minValue), 1);
        minValueMat = minValueMat';
        costMat = distanceMatrix.*blockDiagonalOnesMat + margin*blockDiagonalOnesMat - blockDiagonalOnesMat.*minValueMat;
        costMat = (costMat > 0);
        
        indices = [];
        indexSetOfIJK = [];
        
        for sampleIndex = 1:size(distanceMatrix, 1)
            indices = find(costMat(sampleIndex, :));
            ijkTouple = [sampleIndex * ones(1, length(indices)); indices; classKIndices(sampleIndex) * ones(1, length(indices))];
            indexSetOfIJK = [indexSetOfIJK; ijkTouple'];
        end
        
        %second term of gradient
        dupIndexIJK = indexSetOfIJK;
        loss2 = 0;
        gradientGtTerm2 = gradientGtTerm2 * 0;
        
        %Batch gradient descent
        for t = 1:size(indexSetOfIJK, 1)
            i = indexSetOfIJK(t,1); j = indexSetOfIJK(t,2); k = indexSetOfIJK(t,3);
            [t i j k ];
            dupI = i;
            if FAST
                %Swap i and j
                if (i > j)
                    tmpVar = i; i = j; j = tmpVar;
                end
                index1 = (i - 1) * numberOfDataSamples + j - sum(0: i - 1);
            else
                index1 = (i - 1) * numberOfDataSamples + j ;
            end
            
            % swap i and k
            if FAST
                if (dupI > k)
                    tmpVar = dupI; dupI = k; k = tmpVar;
                end
                index2 = (dupI - 1) * numberOfDataSamples + k - sum(0: dupI - 1);
            else
                index2 = (i - 1) * numberOfDataSamples + k;
            end
            %NOTE***** sign is changed to negative. Original paper positive.
            outerProducts = outerProductMatrix{index1} - outerProductMatrix{index2};
            %NOTE***** sign is changed to negative. Original paper positive.
            gradientGtTerm2 = gradientGtTerm2 + outerProducts;
            %NOTE***** sign is changed to negative. Original paper positive.
            loss2 = loss2 + max(trace(M*outerProducts) + margin, 0);
            %loss2 = loss2 + trace(M*outerProducts);
        end
        
        gradientGt = (1 - alpha) * gradientGtTerm1 + alpha * gradientGtTerm2;
        M = M - lambda * gradientGt;
        tmpM = M;
        M = (M + M')/2;
        all(eig(M) > 0);
        [eigVec eigVal] = eig(M);
        eigVal(eigVal < 0) = 0;
        %M = eigVec * eigVal * inv(eigVec);
        M = eigVec * eigVal * (eigVec)';
        all(eig(M) >= 0);
        %sprintf('Number of non-zero eigen values %d outof %d', nnz(eigVal), size(eigVal, 1))
        totalLoss(iteration + 1) = (1 - alpha) * loss1 + alpha * loss2;
        arrayLoss1(iteration + 1) = loss1;
        arrayLoss2(iteration + 1) = loss2;
        %         if totalLoss(iteration + 1) - totalLoss(iteration) < 0
        %             lambda = 1.1*lambda;
        %         else
        %             lambda = 0.5*lambda;
        %         end
        %             arrayLambda(iteration + 1) = lambda;
        arrayM{iteration} = M;
        iteration = iteration + 1;
    end
end
toc

figure;plot(arrayLoss1(2:end), 'r');hold on;
figure;plot(arrayLoss2(2:end), 'g');hold on;
figure;plot(totalLoss(2:end), 'b');title('Loss1: red, Loss2: green, Total: blue')
hold on;

for p = 2:length(arrayM)
    mae(p - 1) = sum(sum(abs(arrayM{p} - arrayM{p - 1})));
end
figure;
plot(mae, 'c');outputData.metricLearned = M;
outputData.totalLoss = totalLoss;
outputData.arrayM = arrayM;











