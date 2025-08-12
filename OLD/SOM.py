import numpy as np
from minisom import MiniSom

def som_train(data, x=10, y=10, sigma=1, learning_rate= 0.05, iters= 10000, neighborhood_function='bubble'):
    input_len = data.shape[1] # number of features
    som = MiniSom(
        x= x, 
        y= y, 
        input_len=input_len, 
        sigma=sigma, # radius of the different neighbors
        learning_rate=learning_rate, # number of weights adjusted during each iteration
        neighborhood_function=neighborhood_function
    )
    som.random_weights_init(data) # initializes the weights by picking random samples from the data
    som.train_random(data, iters) # train the model by picking random samples from our data
    return som

def som_pred(som_model, data, outlier_percentage):
    model = som_model
    data = data.numpy()
    # for each point in data, find the closest neuron in the feature map 
    # and calculate the distance between the two points
    quantization_errors = np.linalg.norm(model.quantization(data) - data, axis=1)
    # threshold that separates normal data from anomalies
    error_threshold = np.percentile(quantization_errors, 100*(1-outlier_percentage)+5)
    is_anomaly = quantization_errors > error_threshold
    y_pred = np.multiply(is_anomaly, 1) # converts True/False to 1/0
    return y_pred