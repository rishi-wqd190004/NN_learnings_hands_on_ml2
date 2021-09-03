# NN_learnings_hands_on_ml2
This is a learning repository made to learn NN part of the Hands on ML book from O'Reilly
*This repository talks about learning of Neural Networks using keras and tf 2.5 and running natively on m1 macbook air*

**Weights**
Weights are mainly seen like how much of the effect of change of input have on the output.
Also shows the strength to the connectection.
Low weights --> No change in the input
Higer weights --> More effect on the output

**Bias**
Bias can be seen as how far are the prediction from the intended values.
High bias --> Network making less assumptions about the form of the output.
Low bias --> Network making more assumptions about the form of the output.

### Note: This repository will be my learnings from O'Reilly book of hands on ML with sklearn, keras and tf
## Chapter 10 learnings:
*Navigate to the folder ch_10 and read for related chapter learnings in README.md file*


**Keras Youtube**
Samples --> input data
Labels --> target data

Sequential --> Only does something when tf.keras.fit is called.
 - x --> input array
 - y --> target data
 format of both x and y should match
 transform data --> More efficient for model to understand
 MinMaxScaler --> To create a feature range to rescale data

 - Validation set --> Unexposed dataset; subset of training data and is validated ; generalize well on the validation set
    Can be created by two ways:
    - Create seperately from the training data and then pass into the model.fit() as argument 
    - Keras create the validation set

- Infrencing --> model takes what it took during training and uses it on test set
Now here comes the confusion matrix as we don't generally have the labels set of the testing data. If we have we can create a confusion matrix to showcase the difference

- model.to_json --> only need to save the architecture
- model.to_yml() --> saving in yml format
- model.save_weights() --> saving weights (don't forget the weights format)

***Batch Size***
Number of samples processed before the model is updated. 

***No. of epochs***
Number of complete passes through the training dataset.

***Latent Space (latent_dim)***
Simply means a representation of compressed data.
Latent means hidden.
Data compression is one reason to do this. The reason why we use data compression is, we need to find patterns. These patterns help in learning for example 19D data point needs 19 values to define unique points and squishing all that information into a 9D data point. This ***learning*** is simply learning features at each layer edges, angles, etc and attributing a combination of features to a specific output. There is input image and then its compressed or reduced and then needs to reconstruct the compressed image, it must learn to store all relevant features/information and disregard noise. --> Helps in removing unwanted data points and focus only on important features. (Source: https://towardsdatascience.com/understanding-latent-space-in-machine-learning-de5a7c687d8d)
The most fascinating thing here is that this is 'hidden'. Latent space is important for learning data features and for finding simplre representations of data for analysis.
Use our model's decoder to 'generate' data samples by interpolating data in the latent space.
To ***visualize*** the latent space use t-SNE and LLE which takes latent space representation and transforms it into 2D or 3D.

