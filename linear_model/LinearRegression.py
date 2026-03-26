"""
Things : - 

# it is simple linear regression
y = mx + b

y = prediction
m = weight
x = value
b = bias

loss = prediction - real_val
dw = gradient descent of weight
db = gradient descenf of bias



"""

import numpy as np

class SimpleLinearRegression():

    def fit(self,
            X : np.array,
            y : np.array, 
            epochs : int = 100,
            learning_rate : float = 0.0001,
            weights : float = 0,
            bias : float = 0,
            verbose : bool = True
            ):
        """

        Input Param*
        __________
        X = Data to Train 1D array, Dtype = np.array  

        Y = True value a.k.a original prediction 1D array, Dtype = np.array  

        epochs = loop to update weight and bias, Dtype = int and default = 100

        learning_rate = how fast weights should update, Dtype = float, Default = 0.0001  

        weights = enter custom weight | optional
        bias = enter custom bias | optional
        __________________
        >>> Change the learning_rate if output or weights are contains 'e' e.g -1.8038873e+163
        """
        # weights =  bias = 0

        n = len(X)
        
        for e in range(epochs):
            if verbose:
                print(f"Epoch #{e} .............")
            
            dw = db = 0
            for items in range(n):
                pred = X[items] * weights + bias # Calculation prediction, y = mx + b

                loss = pred - y[items] # calculating loss

                dw += loss * X[items]
                db += loss
            
            # avg gradients
            dw = (2/n) * dw
            db = (2/n) * db

            # updating weight and bias
            weights -= learning_rate * dw
            bias -= learning_rate * db

        self.weights = weights
        self.bias = bias

    def get_weight_bias(self) -> np.array:
        """Input = None, 
        O/P Type np.array
        
        Output Format
        arr[0] = weight
        arr[1] = bias
        arr[0] arr[1] = float64 Dtype
        """

        return np.array([self.weights, self.bias])
    
    def predict(self, new_data : np.array) -> np.array:
        """
        Input Format = 1D np.array
        Output Format = 1d np.array
        """
        if len(new_data) == 0:
            raise "Got Empty Array"
        
        prediction = []
        for items in new_data:
            pred = items * self.weights + self.bias
            prediction.append(pred)
        
        return np.array(prediction).round(2)


if __name__ == "__main__":
    Model = SimpleLinearRegression()
    X = [10, 20, 30, 40, 50, 60]
    y = [1, 2, 3, 4, 5, 6]
    Model.fit(X, y, epochs=1000, learning_rate=0.00001)
    print(Model.get_weight_bias())

    print(Model.predict(np.array([70, 80])))