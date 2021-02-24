import MachineLearningCourse.MLUtilities.Image.Convolution2D as Convolution2D
from PIL import Image
import math
import time
from joblib import Parallel, delayed

class BlinkFeaturize(object):
    def __init__(self):
        self.featureSetCreated = False
    
    def CreateFeatureSet(self, xRaw, yRaw, includeEdgeFeatures=True, includeRawPixels=False, includeIntensities=False, intensitiesSampleStride = 2, include3by3=False):
        self.includeEdgeFeatures = includeEdgeFeatures
        self.includeRawPixels = includeRawPixels
        self.includeIntensities = includeIntensities
        self.intensitiesSampleStride = int(intensitiesSampleStride)
        self.include3by3 = include3by3
        self.featureSetCreated = True
        
    def _FeaturizeX(self, xRaw):
        featureVector = []
        
        image = Image.open(xRaw)
        
        xSize = image.size[0]
        ySize = image.size[1]
        numPixels = xSize * ySize

        pixels = image.load()

        if self.includeEdgeFeatures:
            yEdges = Convolution2D.Convolution3x3(image, Convolution2D.SobelY)
            xEdges = Convolution2D.Convolution3x3(image, Convolution2D.SobelX)
            
            avgYEdge = sum([sum([abs(value) for value in row]) for row in yEdges]) / numPixels
            avgXEdge = sum([sum([abs(value) for value in row]) for row in xEdges]) / numPixels
            
            featureVector.append(avgYEdge)
            featureVector.append(avgXEdge)

            if self.include3by3:
                # Divide into 3x3 grid
                split_number = 3
                x_chunk = math.floor(xSize/split_number)
                y_chunk = math.floor(ySize/split_number)

                # For now stop if it cannot be divided evenly
                if (math.floor(xSize/split_number) != xSize/split_number) or math.floor(ySize/split_number) != ySize/split_number:
                    print("Image cannot be split evenly")
                    quit()

                # Find the split boxes
                image_boxes = []
                for i in range(split_number):
                    for j in range(split_number):
                        image_boxes.append((i*x_chunk, j*y_chunk, (i+1)*x_chunk, (j+1)*y_chunk))
            
                # For each box crop, load and calculate the max and avg
                for box in image_boxes:
                    sub_image = image.crop(box)
                    xSize = sub_image.size[0]
                    ySize = sub_image.size[1]
                    numPixels = xSize * ySize
                    # sub_pixels = sub_image.load()

                    yEdges = Convolution2D.Convolution3x3(sub_image, Convolution2D.SobelY)
                    xEdges = Convolution2D.Convolution3x3(sub_image, Convolution2D.SobelX)

                    avgYEdge = sum([sum([abs(value) for value in row]) for row in yEdges]) / numPixels
                    avgXEdge = sum([sum([abs(value) for value in row]) for row in xEdges]) / numPixels
                    maxYEdge = max([max([abs(value) for value in row]) for row in yEdges])
                    maxXEdge = max([max([abs(value) for value in row]) for row in xEdges])
                    
                    # Add to feature vector
                    featureVector.append(avgYEdge)
                    featureVector.append(avgXEdge)
                    featureVector.append(maxYEdge)
                    featureVector.append(maxXEdge)

        if self.includeRawPixels:
            for x in range(xSize):
                for y in range(ySize):
                    featureVector.append(pixels[x,y])

        if self.includeIntensities:
            for x in range(0, xSize, self.intensitiesSampleStride):
                for y in range(0, ySize, self.intensitiesSampleStride):
                    featureVector.append(pixels[x,y]/255.0)

        return featureVector

    def Featurize(self, xSetRaw, verbose = True):
        if not self.featureSetCreated:
            raise UserWarning("Trying to featurize before calling CreateFeatureSet")
        
        if verbose:
            print("Loading & featurizing %d image files..." % (len(xSetRaw)))
        
        
        startTime = time.time()

        # If you don't have joblib installed you can swap these comments
        
        # result = [ self._FeaturizeX(x) for x in xSetRaw ]
        result = Parallel(n_jobs=12)(delayed(self._FeaturizeX)(x) for x in xSetRaw)
    
        endTime = time.time()
        runtime = endTime - startTime
        
        if verbose:
            print("   Complete in %.2f seconds" % (runtime))
        
        return result
