% This function is used to normalise the data.
% Inputs:
% *inData: This is input data matrix of size N x D where N is the number of data points
% and D is the dimension of each data point. Data points are arranged along the rows.
% 
% *inNormType: This is the string giving norm type and can take values as 'l1', 'l2' etc.
% 
% Outputs:
% *outData: This is normalised input data according to specified norm in inNormType.
% This is a matrix of size D x N where normalised data points are arranged along the columns.


function outData = functionNormaliseData(inData, inNormType)

if strcmp(inNormType, 'l2')
    %l2 normalise
    outData  = inData./repmat(sum(inData.^2 ,2),1, size(inData ,2));
    outData(isnan(outData)) = 0;
    outData = outData';
elseif strcmp(inNormType, 'l1')
    %l1 normalise
    outData  = inData./repmat(sum(inData ,2),1, size(inData ,2));
    outData(isnan(outData)) = 0;
    outData = outData';
else
    %none
    outData = inData';
end
