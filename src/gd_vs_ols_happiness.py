import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():

    # !EDIT THIS IF THE FILEPATH NEEDS TO CHANGE 
    filename = "datasets/gdp-vs-happiness.csv"

    X, Y = preprocess_data(filename)
    learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]
    epoch_counts = [100, 500, 1000, 2000, 5000]

    #linear regressoin with OLS
    lr_ols = OLS()
    beta_ols = lr_ols.train(X,Y)
    Y_predict = lr_ols.predict(X,beta_ols)

    # linear regression with gradient descent
    lr_gd = gradient_descent()
    print("Gradient Descent Experiments: ")
    results = []
    for lr in learning_rates:
        for epochs in epoch_counts:
            lr_gd = gradient_descent()
            beta_gd = lr_gd.train(X, Y, learning_rate=lr, epochs=epochs)
            mse = lr_gd.calculate_mse(X, Y, beta_gd)
            results.append((lr, epochs, beta_gd, mse))

            print(f"LR: {lr}, Epochs: {epochs}, MSE: {mse:.6f}, Beta: {beta_gd.flatten()}")

    # Our best result from gradient descent
    best_lr, best_epochs, best_beta, best_mse = min(results, key=lambda x: x[3])

    print(f"\nBest Results: ")
    print(f"Best Gradient Descent - Learning rate: {best_lr}, Epochs: {best_epochs}, MSE: {best_mse:.6f}")
    print(f"Best Gradient Descent Beta: {best_beta.flatten()}")
    print(f"OLS Beta: {beta_ols.flatten()}")

    # Below code is for the graphs:
    X_ = X[...,1].ravel()

    # Output 1 - 8 different regression lines
    fig1, ax1 = plt.subplots()
    fig1.set_size_inches((15,8))

    ax1.scatter(X_, Y, alpha=0.6, color='blue', label='Data Points')

    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    
    selected_results = [
        results[4],  
        results[9],  
        results[12], 
        results[15], 
        results[19], 
        results[20], 
        results[24], 
        results[22]  
    ]

    for i, (lr, epochs, beta, mse) in enumerate(selected_results):
        gd_temp = gradient_descent()
        Y_pred = gd_temp.predict(X, beta)
        ax1.plot(X_, Y_pred, color=colors[i], 
                label=f'GD: LR={lr}, Epochs={epochs}, MSE={mse:.3f}')

    ax1.set_xlabel("GDP per capita")
    ax1.set_ylabel("Happiness")
    ax1.set_title("Multiple Gradient Descent Regression Lines")
    ax1.legend()
    plt.show()

    # Output 2 - best gradient descent vs OLS
    fig2, ax2 = plt.subplots()
    fig2.set_size_inches((15,8))

    ax2.scatter(X_, Y, alpha=0.6, color='blue', label='Data Points')

    # Plot OLS line
    ax2.plot(X_, Y_predict, color='red', linewidth=2, linestyle='dashed', label='OLS Method')

    # Plot best gradient descent line
    Y_predict_best = lr_gd.predict(X, best_beta)
    ax2.plot(X_, Y_predict_best, color='green', linewidth=4, linestyle='dotted', alpha=0.6, 
            label=f'Best Gradient Descent (LR={best_lr}, Epochs={best_epochs})')

    ax2.set_xlabel("GDP per capita")
    ax2.set_ylabel("Happiness")
    ax2.set_title("OLS vs Gradient Descent (Part 1)")
    ax2.legend()
    plt.show()

# Copy the preprocessing steps used for the OLS method
def preprocess_data(filename):

    # import data
    data = pd.read_csv(filename)

    #drop columns that will not be used
    by_year = (data[data['Year']==2018]).drop(columns=["World regions according to OWID","Code"])
    # remove missing values from columns 
    df = by_year[(by_year['Cantril ladder score'].notna()) & (by_year['GDP per capita, PPP (constant 2021 international $)']).notna()]

    #create np.array for gdp and happiness where happiness score is above 4.5
    happiness=[]
    gdp=[]
    for row in df.iterrows():
        if row[1]['Cantril ladder score']>4.5:
            happiness.append(row[1]['Cantril ladder score'])
            gdp.append(row[1]['GDP per capita, PPP (constant 2021 international $)'])

    # Convert to numpy arrays
    input_data = np.array(gdp)
    target_data = np.array(happiness)
    
    # normalize the values
    hmean = np.mean(input_data)
    hstd = np.std(input_data)
    x_normalized = (input_data - hmean) / hstd
    
    X = np.column_stack((np.ones(len(x_normalized)), x_normalized))
    
    # Normalize target
    gmean = np.mean(target_data)
    gstd = np.std(target_data)
    y_train = (target_data - gmean) / gstd

    # arrange in matrix format
    Y = (np.column_stack(y_train)).T
    
    return X, Y

# OLS method for linear regression, copied from the given file so we can use it for comparing the output for the 2nd graph
class OLS():

    def __init__(self) -> None:
        pass

    def train(self, X, Y):
        #compute and return beta
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    
    def predict(self, X_test,beta):

        Y_hat = X_test*beta.T
        return np.sum(Y_hat,axis=1)
    
# Gradient descent method
class gradient_descent():
    def __init__(self) -> None:
        pass

    def train(self, X, Y, learning_rate, epochs):

        # number of samples
        n = X.shape[0]
        
        # Random initialization
        beta = np.random.randn(2, 1)  

        for _ in range(epochs):
            
            # Calculate gradients
            gradients = 2/n * (X.T).dot(X.dot(beta) - Y)
            
            # Update beta
            beta = beta - learning_rate * gradients
        
        return beta
    
    def predict(self, X_test, beta):

        Y_hat = X_test.dot(beta)
        return Y_hat.flatten()
    
    def calculate_mse(self, X, Y, beta):

        predictions = X.dot(beta)
        return np.mean((Y - predictions) ** 2)

if __name__ == "__main__":
    main()