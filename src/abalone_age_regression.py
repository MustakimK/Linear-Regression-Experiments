import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # !EDIT THIS IF THE FILEPATH NEEDS TO CHANGE 
    filepath = "datasets/training_data.csv"

    # Loading and splitting the data
    X, Y, X_train, X_test, Y_train, Y_test, feature_names = load_and_split(filepath)
    
    # Instantiate the class
    lr_ols = linear_regression(X_train, Y_train, X_test, Y_test)

    # Normalize the data
    X_train_norm, Y_train_norm = lr_ols.preprocess_train()
    X_test_norm = lr_ols.preprocess_test()

    # Train the model on normalized data
    beta = lr_ols.train(X_train_norm, Y_train_norm)
    
    # Print beta values
    print("Beta Values:")
    print(f"intercept: {beta[0][0]}")
    for i, feature_name in enumerate(feature_names):
        print(f"Beta {i+1} ({feature_name}): {beta[i+1][0]}")
    
    # Make predictions on normalized data
    Y_predict_train = lr_ols.predict(X_train_norm, beta)
    Y_predict_test = lr_ols.predict(X_test_norm, beta)
    
    # Denormalize predictions for MSE calculation
    Y_predict_train_denorm = Y_predict_train * lr_ols.y_std + lr_ols.y_mean
    Y_predict_test_denorm = Y_predict_test * lr_ols.y_std + lr_ols.y_mean
    
    # Calculate MSE
    train_mse = np.mean((Y_train - Y_predict_train_denorm) ** 2)
    test_mse = np.mean((Y_test - Y_predict_test_denorm) ** 2)
    
    print("\nMSE:")
    print(f"Training Set MSE: {train_mse}")
    print(f"Test Set MSE: {test_mse}")
    
    # Plot our data and results
    plotting(X, Y, lr_ols, feature_names, beta)

class linear_regression:
    
    def __init__(self, x_train, y_train, x_test, y_test):
        self.input_train = np.array(x_train)
        self.target_train = np.array(y_train)
        self.input_test = np.array(x_test)
        self.target_test = np.array(y_test)
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self.beta = None

    def preprocess_train(self):

        # Normalize training features
        self.x_mean = np.mean(self.input_train, axis=0)
        self.x_std = np.std(self.input_train, axis=0)
        x_train_norm = (self.input_train - self.x_mean) / self.x_std
        X = np.column_stack((np.ones(len(x_train_norm)), x_train_norm))
        
        self.y_mean = np.mean(self.target_train)
        self.y_std = np.std(self.target_train)
        y_train_norm = (self.target_train - self.y_mean) / self.y_std
        Y = (np.column_stack(y_train_norm)).T
        
        return X, Y
    
    def preprocess_test(self):

        # Normalize test features
        x_test_norm = (self.input_test - self.x_mean) / self.x_std
        X = np.column_stack((np.ones(len(x_test_norm)), x_test_norm))
        
        return X
    
    def train(self, X, Y):

        # Compute and return beta using OLS
        self.beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        return self.beta
    
    def predict(self, X_test, beta):

        Y_hat = X_test.dot(beta)
        return Y_hat.ravel()
    
def load_and_split(filepath):
    # Load data 
    data = pd.read_csv(filepath)

    feature_columns = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']
    target_column = 'Rings'
    
    # Extract features and target
    X = data[feature_columns].values
    Y = data[target_column].values

    X_train = X[:2000]
    X_test = X[2000:]
    Y_train = Y[:2000]
    Y_test = Y[2000:]
    
    return X, Y, X_train, X_test, Y_train, Y_test, feature_columns

def plotting(X, y, model, feature_names, beta):
    n_features = X.shape[1]
    
    # Create subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i in range(n_features):
        ax = axes[i]
        x_feature = X[:, i]
        
        # Create a grid on this feature
        x_grid = np.linspace(x_feature.min(), x_feature.max(), 200)

        # Matrix where other features are at 0
        X_line = np.zeros((len(x_grid), n_features))
        X_line[:, i] = x_grid
        
        # Normalize the line data using model's normalization params
        X_line_norm = (X_line - model.x_mean) / model.x_std
        X_line_with_intercept = np.column_stack((np.ones(len(X_line_norm)), X_line_norm))
        
        # Predict normalized values
        y_line_norm = model.predict(X_line_with_intercept, beta)
        
        # Denormalize predictions
        y_line = y_line_norm * model.y_std + model.y_mean
        
        # Plot
        ax.scatter(x_feature, y, alpha=0.5, s=10, color='blue')
        ax.plot(x_grid, y_line, color='red', linewidth=2)
        
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel('Rings')
        ax.set_title(f'{feature_names[i]} vs Rings')
        ax.grid(True, alpha=0.3)
    
    # Remove extra plots
    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle('Abalone Age Prediction (Part 2)')
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()