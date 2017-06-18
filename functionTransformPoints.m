function outData = functionTransformPoints(inData, inLabels, inM)

[eigVec eigVal] = eig(inM);
eigVal(eigVal < 10^-8) = 0;
if nnz(eigVal) > 1
    outData.flag = 1;
else
    outData.flag = 0;
end
M_power_half = eigVec * sqrt(eigVal) * inv(eigVec);
data = M_power_half * inData';
outData.data = data';


