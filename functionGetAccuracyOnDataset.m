

function [outAccuracy inferedLabels distanceMat] = functionGetAccuracyOnDataset(inTestData, inGroundTruthLabels, ...
    inTestClassLabels, inPrototypes, inMetric)

inferedLabels = zeros(size(inTestData, 1), 1);
distanceMat = [];
for t = 1:size(inTestData, 1)
    p = 1;
    for class = 1:length(inTestClassLabels)
        xi = inTestData(t, :);
        xj = inPrototypes(:, inTestClassLabels(class))';
        ximxj = xi - xj;
        classDistances(p) = ximxj * inMetric * ximxj';
        p = p + 1;
    end
    distanceMat = [distanceMat; classDistances];
    [minVal index] = min(classDistances);
    inferedLabels(t, 1) = inTestClassLabels(index);
end

%Find accuracy 
outAccuracy = 100 * sum(inGroundTruthLabels == inferedLabels)/length(inferedLabels);