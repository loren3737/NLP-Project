
#Everything
import MachineLearningCourse.MLUtilities.Data.Generators.SampleUniform2D as SampleUniform2D
import MachineLearningCourse.MLUtilities.Data.Generators.ConceptCircle2D as ConceptCircle2D
import MachineLearningCourse.MLUtilities.Data.Generators.ConceptSquare2D as ConceptSquare2D
import MachineLearningCourse.MLUtilities.Data.Generators.ConceptLinear2D as ConceptLinear2D
import MachineLearningCourse.MLUtilities.Data.Generators.ConceptCompound2D as ConceptCompound2D
import MachineLearningCourse.MLProjectSupport.Blink.BlinkDataset as BlinkDataset
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
from PIL import Image
import torchvision.transforms as transforms
import torch 
import MachineLearningCourse.Assignments.Module03.SupportCode.BlinkFeaturize as BlinkFeaturize
import MachineLearningCourse.Assignments.Module03.SupportCode.BlinkNeuralNetwork as BlinkNeuralNetwork

kOutputDirectory = "C:\\temp\\visualize\\lenet"

(xRaw, yRaw) = BlinkDataset.LoadRawData()

(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw)

print("Train is %d samples, %.4f percent opened." % (len(yTrain), 100.0 * sum(yTrain)/len(yTrain)))
print("Validate is %d samples, %.4f percent opened." % (len(yValidate), 100.0 * sum(yValidate)/len(yValidate)))
print("Test is %d samples %.4f percent opened" % (len(yTest), 100.0 * sum(yTest)/len(yTest)))

##
# Load the images, normalize and convert them into tensors
##

featurizer = BlinkFeaturize.BlinkFeaturize()
sampleStride = 1
featurizer.CreateFeatureSet(xTrainRaw, yTrain, includeEdgeFeatures=False, includeIntensities=True, intensitiesSampleStride=sampleStride)

xTrain    = featurizer.Featurize(xTrainRaw)
xValidate = featurizer.Featurize(xValidateRaw)
xTest     = featurizer.Featurize(xTestRaw)

transform = transforms.Compose([
            transforms.ToTensor()
            ,transforms.Normalize(mean=[0.], std=[0.5])
            ])

xTrainImages = [ Image.open(path) for path in xTrainRaw ]
xTrain = torch.stack([ transform(image) for image in xTrainImages ])
yTrain = torch.Tensor([ [ yValue ] for yValue in yTrain ])
xValidateImages = [ Image.open(path) for path in xValidateRaw ]
xValidate = torch.stack([ transform(image) for image in xValidateImages ])
yValidate = torch.Tensor([ [ yValue ] for yValue in yValidate ])
xTestImages = [ Image.open(path) for path in xTestRaw ]
xTest = torch.stack([ transform(image) for image in xTestImages ])
yTest = torch.Tensor([ [ yValue ] for yValue in yTest ])

# Creat the model
model = BlinkNeuralNetwork.BlinkNeuralNetwork(first_convolution = 6, second_convolution = 16, first_connected_nodes = 120, second_connected_nodes = 84, output_nodes = 1)

# Find the graphics card
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device is:", device)


# Move the data onto whichever device was selected
model.to(device)
xTrain = xTrain.to(device)
yTrain = yTrain.to(device)
xValidate = xValidate.to(device)
yValidate = yValidate.to(device)

model.train_model_persample(xTrain, yTrain)
# model.train_model(xTrain, yTrain)

print("Accuracy and Error Bounds:")
yValidatePredicted = model.predict(xValidate)
validAccuracy = EvaluateBinaryClassification.Accuracy(yValidate, yValidatePredicted)
print(validAccuracy)

num_samples = len(xValidate)
(low_bound, high_bound) = ErrorBounds.GetAccuracyBounds(validAccuracy, num_samples, 0.5)
errorBound = (high_bound - low_bound) / 2
print(errorBound)

    # learning_error_series = []
    # learning_valid_series = []
    # learning_series = []

    # converg_error_series = []
    # converg_valid_series = []
    # converg_series = []

    # # log_convert = {0.1: 0, 0.01: 1, 0.001: 2, 0.0001: 3, 0.00001: 4}

    # Charting.PlotSeriesWithErrorBars([converg_valid_series], [converg_error_series], ["Accuracy"], [converg_series], chartTitle="<NN Accuracy on Validation Data>", xAxisTitle="<converg>", yAxisTitle="<Accuracy>", yBotLimit=0.65, outputDirectory=kOutputDirectory, fileName="converg_sweep")
    # Charting.PlotSeriesWithErrorBars([learning_valid_series], [learning_error_series], ["Accuracy"], [learning_series], chartTitle="<NN Accuracy on Validation Data>", xAxisTitle="<learning>", yAxisTitle="<Accuracy>", yBotLimit=0.65, outputDirectory=kOutputDirectory, fileName="learning_sweep")

