import numpy as np
from scipy.stats import truncnorm
from scipy.special import expit as activation_function

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

class NeuralNetwork:
   
    def __init__(self, 
                 no_of_in_nodes, 
                 no_of_out_nodes, 
                 no_of_hidden_nodes,
                 learning_rate,
                 bias=None):
        
        self.no_of_in_nodes = no_of_in_nodes
        
        self.no_of_out_nodes = no_of_out_nodes 
        
        self.no_of_hidden_nodes = no_of_hidden_nodes
        
        self.learning_rate = learning_rate  
        
        self.bias = bias
        
        self.create_weight_matrices()
        
    def create_weight_matrices(self):
        
        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes, 
                                       self.no_of_in_nodes))
        
        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_out = X.rvs((self.no_of_out_nodes, 
                                        self.no_of_hidden_nodes))
           
    
    def train(self,input_vector,target_vector):
        
        input_vector = np.array(input_vector)
        input_vector = input_vector.reshape(input_vector.size,1)
        
        if self.bias:
            input_vector = np.concatenate((input_vector,[[self.bias]]))
        
        target_vector = np.array(target_vector).reshape(target_vector.size,1)
        
        output_vector_hidden = activation_function(self.weights_in_hidden @ input_vector)
        
        if self.bias :
            output_vector_hidden = np.concatenate((output_vector_hidden,[[self.bias]]))
        
        output_vector_netowork = activation_function(self.weights_hidden_out @ output_vector_hidden)
        
        output_error = target_vector -output_vector_netowork

        tmp = output_error *output_vector_netowork*(1-output_vector_netowork)
        
        self.weights_hidden_out += self.learning_rate *(tmp @ output_vector_hidden.T)
        
        
        hidden_error = self.weights_hidden_out.T @ output_error
        
        tmp = hidden_error * output_vector_hidden *(1-output_vector_hidden)
        
        if self.bias:
            
            x= (tmp@ input_vector.T)[:-1,:]
        else:
            x =( tmp@ input_vector.T)
        
        self.weights_in_hidden += self.learning_rate *x
        
                
    
    def run(self, input_vector):
        
        input_vector = np.array(input_vector)
        input_vector = input_vector.reshape(input_vector.size,1)
        
        if self.bias:
            
            input_vector = np.concatenate((input_vector,[[0.02]]))
            
        input4hidden = activation_function(self.weights_in_hidden @ input_vector)
        
        if self.bias:
            
            input4hidden = np.concatenate(input4hidden,[[0.02]])
            
        output_vector_netowork = activation_function(self.weights_hidden_out @ input4hidden)
        
        return output_vector_netowork
    
    def evaluate(self,data,labels):
        
        correct,wrongs =0,0
        for i in range(len(data)):
            
            res = self.run(data[i])
            
            res_max = res.argmax()
            
            if res_max == labels[i].argmax():
                correct =correct+1
            else:
                wrongs =wrongs+1
                
        return print(correct,wrongs)
        
        
from sklearn.datasets import make_blobs

n_samples =500
blob_centers = ([2,6],[6,2],[7,7])

n_classes = len(blob_centers)

data,labels = make_blobs(n_samples =n_samples,
                         centers=blob_centers,
                         random_state=7)

labels = np.arange(n_classes) == labels.reshape(labels.size, 1)
labels = labels.astype(np.float)



from sklearn.model_selection import train_test_split

res = train_test_split(data, labels, 
                       train_size=0.8,
                       test_size=0.2,
                       random_state=42)
train_data, test_data, train_labels, test_labels = res    




simple_network = NeuralNetwork(no_of_in_nodes=2, 
                               no_of_out_nodes=3, 
                               no_of_hidden_nodes=2,
                               learning_rate=0.3)


for i in range(len(train_data)):
    simple_network.train(train_data[i], train_labels[i])

simple_network.evaluate(train_data, train_labels)


for i in range(len(test_data)):
    simple_network.train(test_data[i], test_labels[i])

simple_network.evaluate(test_data, test_labels)

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

multi = MLPClassifier()

multi = multi.fit(train_data,train_labels)

result = multi.predict(test_data)

result1 = multi.predict(train_data)#

print(accuracy_score(result,test_labels))
print(accuracy_score(result1,train_labels))



print(multi.score(train_data,train_labels))
print(multi.score(test_data,test_labels))









