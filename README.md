# Deep Learning For Species Recognition
This is a repository for sharing deep learning algorithms for recognizing species from images (including camera trap and drone images), such as Amur tiger, leopard, roe deer, et al. 

# Model 
(1) Used transfer learning technique;
(2) Based on ResNet50 pre-trained from imagenet dataset;
(3) The last layer of ResNet50 is removed, and then from the second add the following layers:
    (a) average pooling layer 2by2;
    (b) Use relu activation function to add a 1024 neuron fully connected layer; 
    (c) Add an output layer with 3 neurons and softmax activation function.

# Model Training 
Model: optimizer: adam, loss function: 'categorical_crossentropy', metrics: accuracy. 
Epochs: 30, 
Batch_size: 32



