function funtionVisualiseData(inData, inLabels, inClassNames, inNUMBER_OF_CLASSES, inFigureTitle)
    
    labelsAsClassNames = functionGetLabelsAsClassNames(inClassNames, inLabels);
    mappedData = funtionTSNEVisualisation(inData');
    figureTitle = sprintf(inFigureTitle);
    functionMyScatterPlot(mappedData, labelsAsClassNames', inNUMBER_OF_CLASSES, inFigureTitle);
