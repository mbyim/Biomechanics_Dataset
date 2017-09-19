# Biomechanics_Dataset

Neural Networks and Random Forests on Biomechanical Data

Biomehcanics Classifier.ipynb: (Jupyter Notebook) was an attempt to build a multi-class neural network based on the dataset in order to classify the conditions: Normal, Hernia, or Spondylolisthesis. While I was able to build the model using Tensorflow and following some of the work I did in Andrew Ng's Deep Learning specialization, the results were pretty poor. Tinkering with the number of layers and nodes per layer did little to help the accuracy of the model. Given that the dataset contains 310 observations, of which I train on even less, I'm inclined to believe that there is simply too much bias; I don't have enough data to build a strong classifier using a neural network.

random_forest.py: Here, I decided to model the data using Random Forests, which are likely to perform better with less data, and are hard to overfit. Indeed, within the first few tries I was able to get accuracy results upwards of 90% (I didn't keep the random seed) on my dev/CV set! Compared to the poor performance of my neural network, this was much better, and sklearn made it quite straight forward as well. 


