function funtionMySCATTERPLOT(varargin)

inData = varargin{1};
colorMap = hsv(inData.numberOfClasses);
color = [];

for p = 1:inData.numberOfClasses
    color = [color; repmat(colorMap(p, :), inData.numberOfSamplesPerClass, 1)];
end

sz = 50;
figure;
scatter(inData.data(:,1), inData.data(:,2), sz, color, 'filled')
hold on