import MachineLearningCourse.MLProjectSupport.Blink.BlinkDataset as BlinkDataset
#Everything
import MachineLearningCourse.MLUtilities.Data.Generators.SampleUniform2D as SampleUniform2D
import MachineLearningCourse.MLUtilities.Data.Generators.ConceptCircle2D as ConceptCircle2D
import MachineLearningCourse.MLUtilities.Data.Generators.ConceptSquare2D as ConceptSquare2D
import MachineLearningCourse.MLUtilities.Data.Generators.ConceptLinear2D as ConceptLinear2D
import MachineLearningCourse.MLUtilities.Data.Generators.ConceptCompound2D as ConceptCompound2D

import MachineLearningCourse.MLUtilities.Learners.DecisionTreeWeighted as DecisionTreeWeighted
import MachineLearningCourse.MLUtilities.Learners.DecisionTree as DecisionTree
import MachineLearningCourse.Assignments.Module02.SupportCode.AdultFeaturize as AdultFeaturize
import time
import numpy
import MachineLearningCourse.MLProjectSupport.Adult.AdultDataset as AdultDataset
import MachineLearningCourse.MLUtilities.Data.Sample as Sample
import MachineLearningCourse.MLUtilities.Data.CrossValidation as CrossValidationUtil
import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds
import MachineLearningCourse.MLUtilities.Learners.BoostedTree as BoostedTree
import MachineLearningCourse.MLUtilities.Learners.NeuralNetworkFullyConnected as NeuralNetworkFullyConnected

import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
import MachineLearningCourse.MLUtilities.Visualizations.Visualize2D as Visualize2D

(xRaw, yRaw) = BlinkDataset.LoadRawData()

import MachineLearningCourse.MLUtilities.Data.Sample as Sample

(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw)

eyTrain = yTrain
eyValidate = yValidate

print("Train is %d samples, %.4f percent opened." % (len(yTrain), 100.0 * sum(yTrain)/len(yTrain)))
print("Validate is %d samples, %.4f percent opened." % (len(yValidate), 100.0 * sum(yValidate)/len(yValidate)))
print("Test is %d samples %.4f percent opened" % (len(yTest), 100.0 * sum(yTest)/len(yTest)))

from PIL import Image
import torchvision.transforms as transforms
import torch 

kOutputDirectory = "C:\\temp\\visualize\\torch"

import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting

##
# Load the images, normalize and convert them into tensors
##

import MachineLearningCourse.Assignments.Module03.SupportCode.BlinkFeaturize as BlinkFeaturize

featurizer = BlinkFeaturize.BlinkFeaturize()

sampleStride = 2
featurizer.CreateFeatureSet(xTrainRaw, yTrain, includeEdgeFeatures=False, includeIntensities=True, intensitiesSampleStride=sampleStride)

xTrain_pretorch    = featurizer.Featurize(xTrainRaw)
xValidate_pretorch = featurizer.Featurize(xValidateRaw)
xTest_pretorch     = featurizer.Featurize(xTestRaw)

# transform = transforms.Compose([
#             transforms.ToTensor()
#             ,transforms.Normalize(mean=[0.], std=[0.5])
#             ])

# xTrainImages = [ Image.open(path) for path in xTrainRaw ]

# xTrain = torch.stack(torch.FloatTensor(xValue) for xValue in xTrain_pretorch)
xTrain = torch.tensor(xTrain_pretorch)

yTrain = torch.Tensor([ [ yValue ] for yValue in yTrain ])

# xValidateImages = [ Image.open(path) for path in xValidateRaw ]
xValidate = torch.tensor(xValidate_pretorch)

yValidate = torch.Tensor([ [ yValue ] for yValue in yValidate ])

# xTestImages = [ Image.open(path) for path in xTestRaw ]
xTest = torch.tensor(xTest_pretorch)

yTest = torch.Tensor([ [ yValue ] for yValue in yTest ])


######
######

import MachineLearningCourse.Assignments.Module03.SupportCode.BlinkNeuralNetwork as BlinkNeuralNetwork

import time

# Create the loss function to use (Mean Square Error)


# Create the optimization method (Stochastic Gradient Descent) and the step size (lr -> learning rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.008)

##
# Move the model and data to the GPU if you're using your GPU
##

dosweep = True
if dosweep:

    def ExecuteFitting(runSpecification, xTrain, yTrain, xValidate, yValidate):
        startTime = time.time()

        # Create features and train based on type of model
        # Create the model
        model = BlinkNeuralNetwork.BlinkNeuralNetwork(hiddenNodes = 6, hiddenNodesTwo = 4)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device is:", device)

        model.to(device)

        # Move the data onto whichever device was selected
        xTrain = xTrain.to(device)
        yTrain = yTrain.to(device)
        xValidate = xValidate.to(device)
        yValidate = yValidate.to(device)
        
        converged = False
        epoch = 1
        lastLoss = None
        convergence = runSpecification['convergence']
        optimizer = torch.optim.SGD(model.parameters(), lr=runSpecification['learning_rate'])
        lossFunction = torch.nn.MSELoss(reduction='mean')
        patience = 0

        while not converged and epoch < 5000:
            # Do the forward pass
            yTrainPredicted = model(xTrain)
            trainLoss = lossFunction(yTrainPredicted, yTrain)

            # Reset the gradients in the network to zero
            optimizer.zero_grad()

            # Backprop the errors from the loss on this iteration
            trainLoss.backward()

            # Do a weight update step
            optimizer.step()

            loss = trainLoss.item()
            # print(loss)
            if epoch > 10 and lastLoss != None and abs(lastLoss - loss) < convergence:
                if patience >=  0:
                    converged = True
                    pass
                else: 
                    patience += 1
            else:
                lastLoss = loss
                patience = 0
                
            epoch = epoch + 1

        model.train(mode=True)

        endTime = time.time()

        runSpecification['runtime'] = endTime - startTime
        runSpecification['epoch'] = epoch
        
        yValidatePredicted = model(xValidate)
        validAccuracy = EvaluateBinaryClassification.Accuracy(yValidate, [ 1 if pred > 0.5 else 0 for pred in yValidatePredicted ])
        runSpecification['accuracy'] = validAccuracy

        num_samples = len(xValidate)
        (low_bound, high_bound) = ErrorBounds.GetAccuracyBounds(validAccuracy, num_samples, 0.5)
        errorBound = (high_bound - low_bound) / 2
        runSpecification['50PercentBound'] = errorBound
        
        return runSpecification
    
    evaluationRunSpecifications = []


    # for convergence in [0.0005, 0.0001, 0.00001, 0.000001, 0.0000001]:
    #     for learning in [0.05, 0.005, 0.0008, 0.0006, 0.0005, 0.0004, 0.0002, 0.0001, 0.00008]:
    for convergence in [0.1, 0.01, 0.001, 0.0001, 0.00006, 0.00003, 0.00001, 0.000005, 0.000001]:
        for learning in [200, 100, 20, 10, 2, 1, 0.1, 0.01, 0.008, 0.005, 0.001, 0.001]:
            runSpecification = {}

            runSpecification['model'] = 'NN'
            runSpecification['optimizing'] = 'learning_rate'
            runSpecification['learning_rate'] = learning
            runSpecification['convergence'] = convergence
                    
            evaluationRunSpecifications.append(runSpecification)

    # for convergence in [0.1, 0.05, 0.01, 0.005, 0.001]:
    #     runSpecification = {}

    #     runSpecification['model'] = 'NN'
    #     runSpecification['optimizing'] = 'convergence'
    #     runSpecification['learning_rate'] = 0.001
    #     runSpecification['convergence'] = convergence
                
    #     evaluationRunSpecifications.append(runSpecification)

    ## if you want to run in parallel you need to install joblib as described in the lecture notes and adjust the comments on the next three lines...
    from joblib import Parallel, delayed

    # evaluations = Parallel(n_jobs=12)(delayed(ExecuteFitting)(runSpec, xTrain, yTrain, xValidate, yValidate) for runSpec in evaluationRunSpecifications)
    evaluations = [ ExecuteFitting(runSpec, xTrain, yTrain, xValidate, yValidate) for runSpec in evaluationRunSpecifications ]

    for evaluation in evaluations:
        print(evaluation)

    # Produce a plot showing the values of the parameter being optimized on X
    # and accuracy with 50 Percent error bars on Y

    learning_error_series = []
    learning_valid_series = []
    learning_series = []

    converg_error_series = []
    converg_valid_series = []
    converg_series = []

    # log_convert = {0.1: 0, 0.01: 1, 0.001: 2, 0.0001: 3, 0.00001: 4}

    for evaluation in evaluations:
        if evaluation["optimizing"] == "convergence":
            converg_error_series.append(evaluation['50PercentBound'])
            converg_valid_series.append(evaluation['accuracy'])
            converg_series.append(evaluation['convergence'])
        elif evaluation["optimizing"] == "learning_rate":
            learning_error_series.append(evaluation['50PercentBound'])
            learning_valid_series.append(evaluation['accuracy'])
            learning_series.append(evaluation['learning_rate'])
        

    Charting.PlotSeriesWithErrorBars([converg_valid_series], [converg_error_series], ["Accuracy"], [converg_series], chartTitle="<NN Accuracy on Validation Data>", xAxisTitle="<converg>", yAxisTitle="<Accuracy>", yBotLimit=0.65, outputDirectory=kOutputDirectory, fileName="converg_sweep")
    Charting.PlotSeriesWithErrorBars([learning_valid_series], [learning_error_series], ["Accuracy"], [learning_series], chartTitle="<NN Accuracy on Validation Data>", xAxisTitle="<learning>", yAxisTitle="<Accuracy>", yBotLimit=0.65, outputDirectory=kOutputDirectory, fileName="learning_sweep")


roccompare = False
if roccompare:
    
    # A helper function for calculating FN rate and FP rate across a range of thresholds
    def TabulateModelPerformanceForROC(model, xValidate, yValidate):
        pointsToEvaluate = 100
        thresholds = [ x / float(pointsToEvaluate) for x in range(pointsToEvaluate + 1)]
        FPRs = []
        FNRs = []

        try:
            for threshold in thresholds:
                FPRs.append(EvaluateBinaryClassification.FalsePositiveRate(yValidate, model.predict(xValidate, classificationThreshold=threshold)))
                FNRs.append(EvaluateBinaryClassification.FalseNegativeRate(yValidate, model.predict(xValidate, classificationThreshold=threshold)))
        except NotImplementedError:
            raise UserWarning("The 'model' parameter must have a 'predict' method that supports using a 'classificationThreshold' parameter with range [ 0 - 1.0 ] to create classifications.")

        return (FPRs, FNRs, thresholds)

     # Gather ROC curves
    FNRs_series = []
    FPRs_series = []
    Label_series = []

    ####Edge features only model
    print("Starting with just edge features...")
 
    # Create features and train based on type of model
    # Create the model
    model = BlinkNeuralNetwork.BlinkNeuralNetwork(hiddenNodes = 6, hiddenNodesTwo = 4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device is:", device)

    model.to(device)

    # Move the data onto whichever device was selected
    xTrain = xTrain.to(device)
    yTrain = yTrain.to(device)
    xValidate = xValidate.to(device)
    yValidate = yValidate.to(device)
    
    converged = False
    epoch = 1
    lastLoss = None
    convergence = 0.0001
    optimizer = torch.optim.SGD(model.parameters(), lr=0.008)
    lossFunction = torch.nn.MSELoss(reduction='sum')
    patience = 0

    while not converged and epoch < 5000:
        # Do the forward pass
        yTrainPredicted = model(xTrain)
        trainLoss = lossFunction(yTrainPredicted, yTrain)

        # Reset the gradients in the network to zero
        optimizer.zero_grad()

        # Backprop the errors from the loss on this iteration
        trainLoss.backward()

        # Do a weight update step
        optimizer.step()
        
        loss = trainLoss.item() / len(xTrain)
        if epoch > 10 and lastLoss != None and abs(lastLoss - loss) < convergence:
            if patience >  4:
                converged = True
                pass
            else: 
                patience += 1
        else:
            lastLoss = loss
            patience = 0
            
        epoch = epoch + 1

    model.train(mode=True)

    # Check accuracies
    torch_accuracy = EvaluateBinaryClassification.Accuracy(yTrain,model.predict(xTrain))
    print("Training accuracy: " + str(torch_accuracy))
    torch_accuracy = EvaluateBinaryClassification.Accuracy(yValidate,model.predict(xValidate))
    print("Validation accuracy: " + str(torch_accuracy))

    # Calculate ROC curve
    (modelFPRs, modelFNRs, thresholds) = TabulateModelPerformanceForROC(model, xValidate, yValidate)

    FNRs_series.append(modelFNRs)
    FPRs_series.append(modelFPRs)
    Label_series.append("PyTorch")

    #### Include 3x3 Grid Features
    print("Moving on to 3x3 features...")
 
    # Featureize
    ex_featurizer = BlinkFeaturize.BlinkFeaturize()
    ex_featurizer.CreateFeatureSet(xTrainRaw, yTrain, includeEdgeFeatures=False, includeIntensities=True, intensitiesSampleStride=sampleStride)

    exTrain    = ex_featurizer.Featurize(xTrainRaw)
    exValidate = ex_featurizer.Featurize(xValidateRaw)
    exTest     = ex_featurizer.Featurize(xTestRaw)

    # Create the model
    hiddenStructure = [6, 4]
    maxEpochs = 1000
    step = 0.1
    convergence = 0.00001
    
    ex_model = NeuralNetworkFullyConnected.NeuralNetworkFullyConnected(len(exTrain[0]), hiddenLayersNodeCounts=hiddenStructure)
    ex_model.fit(exTrain, eyTrain, maxEpochs = 5000, stepSize=step, convergence=convergence)

    # Check accuracies
    feat_accuracy = EvaluateBinaryClassification.Accuracy(eyTrain,ex_model.predict(exTrain))
    print("Training accuracy: " + str(feat_accuracy))
    feat_accuracy = EvaluateBinaryClassification.Accuracy(eyValidate,ex_model.predict(exValidate))
    print("Validation accuracy: " + str(feat_accuracy))

    # Calculate ROC curve
    (modelFPRs, modelFNRs, thresholds) = TabulateModelPerformanceForROC(ex_model, exValidate, eyValidate)

    FNRs_series.append(modelFNRs)
    FPRs_series.append(modelFPRs)
    Label_series.append("My NN Model")
    
    # Plot ROC curve
    Charting.PlotROCs(FPRs_series, FNRs_series, Label_series, useLines=True, chartTitle="ROC Comparison", xAxisTitle="False Negative Rate", yAxisTitle="False Positive Rate", outputDirectory=kOutputDirectory, fileName="FinalROCCurve")

   