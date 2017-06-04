%% Function to precompute kernel matrix
%
%   Inputs:
%   func_PrecomputeKernel(A,B,K) - A and B are feature matrices
%   respectively. K is the kernel type as a string.
%   A and B are of N x D where N is number of data points and D is feature 
%   dimension. Data points are stored along rows.
%

function [D,gamma] = func_PrecomputeKernel(varargin)

% addpath('/import/geb-experiments/Alex/CVPR15/Code/Zeroshot/HMDBregression/Code/Basic/PrecomputeKernel');

Kernel = lower(varargin{3});
A = varargin{1};
B = varargin{2};

gamma = [];

if nargin >= 4
    gamma = varargin{4};
end

switch Kernel
    case 'chisq'
        D = chi2Kernel(A,B);
    case 'cosine'
        D = 1-pdist2(A,B,'cosine');
    case 'linear'
        D = A * B';
    case 'rbf'
        distM = pdist2(A,B).^2;
        
        if isempty(gamma)
            gamma = 1/mean(mean(distM));    % normalizer for RBF kernel
        end
        D = exp(-gamma * distM);
    case 'rbfchisq'
        D = 1-chi2Kernel(A,B);
        
        if isempty(gamma)
            gamma = 1/mean(mean(D));    % normalizer for RBF kernel
        end
        D = exp(-D*gamma);

end
    