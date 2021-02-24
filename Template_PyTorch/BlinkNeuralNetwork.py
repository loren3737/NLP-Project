import torch
import math

class BlinkNeuralNetwork(torch.nn.Module):
    def __init__(self, first_convolution = 6, second_convolution = 16, first_connected_nodes = 120, second_connected_nodes = 84, output_nodes = 1, 
                    drop_prob = 0.2, in_dropout = [False, False, False, False], in_batch = [False, False, False, False]):
        super(BlinkNeuralNetwork, self).__init__()

        # Tracked information
        self.total_epochs = 0

        # Set convolution and sampling hyper-parameters
        self.convolution_size = 5
        self.convolution_stride = 1
        self.pooling_size = 2
        self.pooling_stride = 2
        self.image_size = 24

        self.input_channel = 1

        # Extra layer positions [A, B, C, D]
        self.dropout_position = in_dropout
        self.batch_position = in_batch

        # Sampling Layer
        self.sampling = torch.nn.AvgPool2d(
            kernel_size = self.pooling_size, 
            stride = self.pooling_stride)

        # Batch Normalization 2D
        batch_features_a = first_convolution
        self.batch_nomalize_a = torch.nn.BatchNorm2d(
            batch_features_a
        )

        batch_features_b = second_convolution
        self.batch_nomalize_b = torch.nn.BatchNorm2d(
            batch_features_b
        )

        # Batch Normalization 1D
        batch_features_c = first_connected_nodes
        self.batch_nomalize_c = torch.nn.BatchNorm1d(
            batch_features_c
        )

        batch_features_d = second_connected_nodes
        self.batch_nomalize_d = torch.nn.BatchNorm1d(
            batch_features_d
        )

        # Dropout Layer
        self.dropout = torch.nn.Dropout(p = drop_prob)

        # Convolution of input features
        self.convolutionOne = torch.nn.Sequential(
            torch.nn.Conv2d(self.input_channel, first_convolution, self.convolution_size, stride=self.convolution_stride),
            torch.nn.ReLU()
        )

        # Convolution
        self.convolutionTwo = torch.nn.Sequential(
            torch.nn.Conv2d(first_convolution, second_convolution, self.convolution_size, stride=self.convolution_stride),
            torch.nn.ReLU()
        )

        # Fully connected layer from all the down-sampled input pixels to all the hidden nodes
        # Calculate the amount of features to fully connect
        convolution_pixel_loss = 2 * math.floor(self.convolution_size / 2)
        size_before_connected = (((self.image_size - convolution_pixel_loss) / 2) - convolution_pixel_loss) / 2
        connect_input_size = int(second_convolution * size_before_connected ** 2)

        self.fullyConnectedOne = torch.nn.Sequential(
           torch.nn.Linear(connect_input_size, first_connected_nodes),
           torch.nn.ReLU()
           )

        self.fullyConnectedTwo = torch.nn.Sequential(
           torch.nn.Linear(first_connected_nodes, second_connected_nodes),
           torch.nn.ReLU()
           )

        # Fully connected layer from the hidden layer to a single output node
        self.outputLayer = torch.nn.Sequential(
            torch.nn.Linear(second_connected_nodes, output_nodes),
            torch.nn.Sigmoid()
            )

    def forward(self, x):
        # Apply the layers created at initialization time in order
        
        # Perform first convolution and sampling layers
        out = self.convolutionOne(x)
        out = self.sampling(out)

        # Extra layer A
        if self.dropout_position[0]:
            out = self.dropout(out)
        if self.batch_position[0]:
            out = self.batch_nomalize_a(out)

        # Perform second convolution and sampling layers
        out = self.convolutionTwo(out)
        out = self.sampling(out)

        # Extra layer B
        if self.dropout_position[1]:
            out = self.dropout(out)
        if self.batch_position[1]:
            out = self.batch_nomalize_b(out)

        # Resize for 1D layers
        out = out.reshape(out.size(0), -1)

        # Perform fully connected layers
        out = self.fullyConnectedOne(out)

        # Extra layer C
        if self.dropout_position[2]:
            out = self.dropout(out)
        if self.batch_position[2]:
            out = self.batch_nomalize_c(out)

        out = self.fullyConnectedTwo(out)

        # Extra layer D
        if self.dropout_position[3]:
            out = self.dropout(out)
        if self.batch_position[3]:
            out = self.batch_nomalize_d(out)
        
        out = self.outputLayer(out)

        # Return output
        return out

    def predict(self, x, classificationThreshold=0.5):
        # Get the ouput propagations
        yValidatePredicted = self.forward(x)
        return [ 1 if pred > classificationThreshold else 0 for pred in yValidatePredicted ]
    
    def train_model(self, xTrain, yTrain, max_epoch = 20000, convergence = 0.000001, learning_rate = 0.1, min_epochs = 10000, patience = 0):
        
        #Set training mode
        self.train(mode=False)    

        # Setup training values
        converged = False
        epoch = 1
        lastLoss = None
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        lossFunction = torch.nn.MSELoss(reduction='mean')
        converge_count = 0

        while not converged and epoch < max_epoch:
            # Do the forward pass
            # for i in range(len(xTrain)):
            yTrainPredicted = self(xTrain)
            trainLoss = lossFunction(yTrainPredicted, yTrain)

            # Reset the gradients in the network to zero
            optimizer.zero_grad()

            # Backprop the errors from the loss on this iteration
            trainLoss.backward()

            # Do a weight update step
            optimizer.step()

            loss = trainLoss.item()
            print("Current Loss: " + str(loss))
            # print(loss)
            if epoch > min_epochs and lastLoss != None and abs(lastLoss - loss) < convergence:
                if converge_count >=  patience:
                    converged = True
                    pass
                else: 
                    converge_count += 1
            else:
                lastLoss = loss
                converge_count = 0
                
            epoch = epoch + 1

        
        print("Total Epochs: " + str(epoch))
        self.total_epochs = epoch
        self.train(mode=True)

    def train_model_persample(self, xTrain, yTrain, max_epoch = 5000, convergence = 0.0001, learning_rate = 0.01, min_epochs = 10, patience = 3):
        
        #Set training mode
        self.train(mode=False)    

        # Setup training values
        converged = False
        epoch = 1
        lastLoss = None
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        # lossFunction = torch.nn.MSELoss()
        lossFunction = torch.nn.MSELoss(reduction='mean')
        converge_count = 0

        while not converged and epoch < max_epoch:
            # Do the forward pass
            for i in range(len(xTrain)):            
                sample = xTrain[i, :, :, :].unsqueeze(0)
                yTrainPredicted = self(sample)
                trainLoss = lossFunction(yTrainPredicted, yTrain[i])

                # Reset the gradients in the network to zero
                optimizer.zero_grad()

                # Backprop the errors from the loss on this iteration
                trainLoss.backward()

                # Do a weight update step
                optimizer.step()

            loss = trainLoss.item()
            print("Current Loss: " + str(loss))
            # print(loss)
            if epoch > min_epochs and lastLoss != None and abs(lastLoss - loss) < convergence:
                if converge_count >=  patience:
                    converged = True
                    pass
                else: 
                    converge_count += 1
            else:
                lastLoss = loss
                converge_count = 0
                
            epoch = epoch + 1

        
        print("Total Epochs: " + str(epoch))
        self.train(mode=True)
        
