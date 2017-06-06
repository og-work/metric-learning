% This function calculates distance between a pair of data points.
% Inputs:
% *inData:
% This is N x D data matrix with N data points of D dimension each. Data arranged along rows
% *inDistMetric: Type of metric e.g. l2, cosine distance etc.
% *M:
% Mahalanobis matrix of dimension D x D
% Outputs:
% outDistanceMatrix:
% This is a matrix of size N x N where entry at (i, j) represents distance
% between ith and jth data point


function outDistanceMatrix = functionGetDistancesBetweenSamples(varargin)

inData1 = varargin{1};
inData2 = varargin{2};
inDistMetric = varargin{3};
M = varargin{4};    

if ~isequal(size(inData1, 2), size(inData2, 2))
    error('Data dimensions not equal')
end

dataDim = size(inData1, 2);
outDistanceMatrix = zeros(size(inData1, 1), size(inData2, 1));
tic

if strcmp(inDistMetric, 'l2')
    for p = 1:size(inData1, 1)
        for q = 1:size(inData2, 1)
            outDistanceMatrix(p, q) = sum((inData1(p, :) - inData2(q, :)).^2, 2);
        end
    end
else
    for p = 1:size(inData1, 1)
        for q = 1:size(inData2, 1)
            outDistanceMatrix(p, q) = (inData1(p, :) - inData2(q, :))* M *(inData1(p, :) - inData2(q, :))';
        end
    end
end

toc
