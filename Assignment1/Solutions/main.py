# ------------------------------------------ Linear Regression ------------------------------------------------
#   Author: Nandhakishore C S 
#   Roll Number: DA24M011
#   Submitted as part of DA5400 Foundation of Machine Learning Assignment 1 

# ----------------------------------------- Importing Libraries -----------------------------------------------

# Matrix Operations 
from numpy import *                 # type: ignore 
import numpy as np                  # type: ignore

# For Plotting Graphs 
import matplotlib.pyplot as plt     # type: ignore 

# ----------------------------- Importing self defined classes and functions ----------------------------------
from data_loader import *       
from linear_regression import * 
from kernel_regression import *

# ---------------------------------------- Main Function ------------------------------------------------------
if __name__ == '__main__': 
    print('\n')
    print(f"{'----------------------------------------------------------------------------------------------------------------': ^150}")

    print(f"{'DA5400 Foundation of Machine Learning' : ^150}")
    print(f"{'Assignment 1' : ^150}")
    print(f"{'Submitted by: Nandhakishore C S' : ^150}")

    print(f"{'----------------------------------------------------------------------------------------------------------------': ^150}")

    print(f"{'Data Loading and Checking for Missing Values' : ^150}")
    print(f"{'Column names are added for ease of accessing the elements' : ^150}")

    # ----------------------------- DATA FOR TRAINING ----------------------------------
    file_path = '/Users/nandhakishorecs/Documents/IITM/Jul_2024/DA5400/Assignments/Assignment1/Dataset/FMLA1Q1Data_train.csv'
    df_train = data_loader(file_path = file_path, title = 'Training')

    X_train = np.array(df_train[['x1', 'x2']])
    y_train = np.array(df_train[['y']])

    # ----------------------------- DATA FOR TESTING -----------------------------------

    file_path = '/Users/nandhakishorecs/Documents/IITM/Jul_2024/DA5400/Assignments/Assignment1/Dataset/FMLA1Q1Data_test.csv'
    df_test = data_loader(file_path = file_path, title = 'Testing')

    X_test = np.array(df_test[['x1', 'x2']])
    y_test = np.array(df_test[['y']])

    print(f"{'----------------------------------------------------------------------------------------------------------------': ^150}")

    # ----------------------------- Model Initialisation -----------------------------------

    model = linear_model() 

    # ----------------------------- (i) Analytical Solution ------------------------------------

    print(f"{'(i) Analytical Solution for Linear Regression - without bias' : ^150}")

    model.analytical_solution(X_train, y_train, fit_intercept = False, alpha = 0)

    print('\nAnalytical Solution for train data:\n', model.parameters(fit_intercept = False))
    predictions = model.predict(X_train, fit_intercept = False)
    print('\nSSE:\t', model.SSE(y_train, predictions))
    print('MSE:\t', model.MSE(y_train, predictions), '\n')

    print(f"{'(i) Analytical Solution for Linear Regression - with bias' : ^150}")

    model.analytical_solution(X_train, y_train, fit_intercept = True, alpha = 0)

    print('\nAnalytical Solution for train data:\n', model.parameters(fit_intercept = True))
    predictions = model.predict(X_train, fit_intercept = True)
    print('\nSSE:\t', model.SSE(y_train, predictions))
    print('MSE:\t', model.MSE(y_train, predictions), '\n')

    print(f"{'----------------------------------------------------------------------------------------------------------------': ^150}")

    # ----------------------------- (ii) Gradient Descent Solution -----------------------------

    print(f"{'(ii) (a) Gradient Descent Solution for Linear Regression - without bias' : ^150}")

    answer = model.fit(X_train, y_train, fit_intercept = False)
    print('\nGD solution of w:\n', model.parameters(fit_intercept = False))
    predictions = model.predict(X_train, fit_intercept = False)
    print('\nSSE:\t', model.SSE(y_train, predictions))
    print('MSE:\t', model.MSE(y_train, predictions), '\n')

    # Plotting iterations(t) vs norm(w_ml - w_t)
    x = np.linspace(start = 0, stop = model._iters, num = len(answer))
    plt.plot(x, answer, 'r-')
    plt.title('Gradient Descent solution without bias')
    plt.suptitle('Euclidean distance between $ w^{t}$ and $w_{MLE}$')
    plt.xlabel('Iterations (t)')
    plt.ylabel('$ {|| w^{t} - w_{MLE} ||}_2 $')
    plt.grid()
    plt.savefig("question_2_(i).png")
    plt.show()
    plt.close()

    print('\n')
    print(f"{'(ii) (b) Gradient Descent Solution for Linear Regression - with bias' : ^150}")

    answer = model.fit(X_train, y_train, fit_intercept = True)
    print('\nGD solution of w:\n', model.parameters(fit_intercept = True))
    predictions = model.predict(X_train, fit_intercept = True)
    print('\nSSE:\t', model.SSE(y_train, predictions))
    print('MSE:\t', model.MSE(y_train, predictions), '\n')

    # Plotting iterations(t) vs norm(w_ml - wt)
    x = np.linspace(start = 0, stop = model._iters, num = len(answer))
    plt.plot(x, answer, 'r-')
    plt.title('Gradient Descent solution with bias')
    plt.suptitle('Euclidean distance between $ w^{t}$ and $w_{MLE}$')
    plt.xlabel('Iterations (t)')
    plt.ylabel('$ {|| w^{t} - w_{MLE} ||}_2 $')
    plt.grid()
    plt.savefig('question_2_(ii).png')
    plt.show()
    plt.close()

    print(f"{'----------------------------------------------------------------------------------------------------------------': ^150}")

    # ----------------------------- (iii) Stochastic Gradient Solution ------------------------------

    print(f"{'(iii) (a) Stochastic Gradient Descent Solution for Linear Regression - without bias' : ^150}")

    answer = model.SGD(X_train, y_train, fit_intercept = False, batch_size = 100)
    print('\nSGD solution of w:\n', model.parameters(fit_intercept = False))
    predictions = model.predict(X_train, fit_intercept = False)
    print('\nSSE:\t', model.SSE(y_train, predictions))
    print('MSE:\t', model.MSE(y_train, predictions), '\n')

    # Plotting iterations(t) vs norm(w_ml - wt)
    x = np.linspace(start = 0, stop = model._iters, num = len(answer))
    plt.plot(x, answer, 'r-')
    plt.title('Stochastic Gradient Descent solution without bias')
    plt.suptitle('Euclidean distance between $ w^{t}$ and $w_{MLE}$')
    plt.xlabel('Iterations (t)')
    plt.ylabel('$ {|| w^{t} - w_{MLE} ||}_2 $')
    plt.grid()
    plt.savefig('question_3_(i).png')
    plt.show()
    plt.close()

    print(f"{'(iii) (b) Stochastic Gradient Descent Solution for Linear Regression - with bias' : ^150}")

    answer = model.SGD(X_train, y_train, fit_intercept = True, batch_size = 100)
    print('\nSGD solution of w:\n', model.parameters(fit_intercept = True))
    predictions = model.predict(X_train, fit_intercept = True)
    print('\nSSE:\t', model.SSE(y_train, predictions))
    print('MSE:\t', model.MSE(y_train, predictions), '\n')

    # Plotting iterations(t) vs norm(w_ml - wt)
    x = np.linspace(start = 0, stop = model._iters, num = len(answer))
    plt.plot(x, answer, 'r-')
    plt.title('Stochastic Gradient Descent solution with bias')
    plt.suptitle('Euclidean distance between $ w^{t}$ and $w_{MLE}$')
    plt.xlabel('Iterations (t)')
    plt.ylabel('$ {|| w^{t} - w_{MLE} ||}_2 $')
    plt.grid() 
    plt.savefig('question_3_(ii).png')
    plt.show() 
    plt.close()

    print(f"{'----------------------------------------------------------------------------------------------------------------': ^150}")

    # -------- (iv) Cross Validation for Ridge Regression using Gradient Descent Solution ----------------
    print(
        f"{'(iv) Cross validation for Regularisation Constant in Ridge Regression solved using Gradient Descent Method' : ^150}"
    )

    lambdas = [0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]

    # Doing 5 fold cross validation 
    errors = model.cross_validate(X_train, y_train, lambdas, split_size = 5)

    # plt.plot(lambdas, errors, 'r-')
    plt.plot(lambdas, errors, 'r-')
    plt.suptitle(r'Cross Validation of Regularisation Constant ($ \lambda $)')
    plt.title('For Ridge Regression solved using Gradient Descent')
    plt.xlabel(r'Regularisation Constants ($ \lambda $)')
    plt.ylabel('Mean Squared Error on Validation data')
    plt.grid() 
    plt.savefig('question_4.png')
    plt.show() 
    plt.close()

    print(
        r'From the graph: Cross Validation of Regularisation Constant, the regularisation constant with minimal error is :\t',
        lambdas[1]
    )

    # Checking Test Error for analytical solution
    model.analytical_solution(X_train, y_train, fit_intercept = False, alpha = 0)
    print('\nAnalytical Solution for test data:\n', model.parameters(fit_intercept = False))
    predictions = model.predict(X_test, fit_intercept = False)
    print('\nSSE for analytical solution on test data:\t', model.SSE(y_test, predictions))
    w_mle_error = model.MSE(y_test, predictions)
    print('MSE for analytical solution on test data:\t', model.MSE(y_test, predictions), '\n')
    
    # Checking Test Error for choosen regularisation constant using gradient descent solution
    answer = model.fit(X_train, y_train, fit_intercept = False, alpha = lambdas[1])
    print('Gradient Descent solution for test data:\n', model.parameters(fit_intercept = False))
    predictions = model.predict(X_test, fit_intercept = False)
    print('\nSSE:\t', model.SSE(y_test, predictions))
    w_r_error = model.MSE(y_test, predictions)
    print('MSE:\t', model.MSE(y_test, predictions), '\n')

    # Ridge regression error is more than MLE error - for report

    print(f"{'----------------------------------------------------------------------------------------------------------------': ^150}")

    # ----------------------------- (V) Kernel Regression ------------------------------

    print(f"{'(v) Solution using Kernel Regression' : ^150}")

    print('\nFor solving the problem, I have taken Radial Basis Kernel, Linear Kernel and Quadratic Kernel')

    # Radial basis Kernel

    # without L2 regularisation
    print('Regression with Radial Basis Function without L2 regularisation:\n')
    K_reg = kernel_regression(alpha = 0)
    predictions = K_reg.fit(X_train, y_train, X_test, 'rbf_kernel', gamma=0.5)
    print('SSE for Radial Basis Kernel:\t', model.SSE(y_test, predictions))
    print('MSE for Radial Basis Kernel:\t', model.MSE(y_test, predictions))

    # with L2 regularisation
    print('\nRegression with Radial Basis Function with L2 regularisation:\n')
    K_reg = kernel_regression(alpha = 0.1)
    predictions = K_reg.fit(X_train, y_train, X_test, 'rbf_kernel', gamma=0.5)
    print('\nSSE for Radial Basis Kernel:\t', model.SSE(y_test, predictions))
    print('MSE for Radial Basis Kernel:\t', model.MSE(y_test, predictions))

    # Linear Kernel

    # without L2 regularisation
    print('\nRegression with Linear Kernel without L2 regularisation:\n')
    K_reg = kernel_regression(alpha = 0)
    predictions = K_reg.fit(X_train, y_train, X_test, 'linear_kernel')
    print('\nSSE for Linear Kernel:\t', model.SSE(y_test, predictions))
    print('MSE for Linear Kernel:\t', model.MSE(y_test, predictions))

    # with L2 regularisation
    print('\nRegression with Linear Kernel with L2 regularisation:\n')
    K_reg = kernel_regression(alpha = 0.1)
    predictions = K_reg.fit(X_train, y_train, X_test, 'linear_kernel')
    print('\nSSE for Linear Kernel:\t', model.SSE(y_test, predictions))
    print('MSE for Linear Kernel:\t', model.MSE(y_test, predictions))

    # Quadratic Kernel - degree 2 version of polynomial kernel

    # without L2 regularisation
    print('\nRegression with Quadratic kernel without L2 regularisation:\n')
    K_reg = kernel_regression(alpha = 0)
    predictions = K_reg.fit(X_train, y_train, X_test, 'quadratic_kernel')
    print('\nSSE for Quadratic Kernel:\t', model.SSE(y_test, predictions))
    print('MSE for Quadratic Kernel:\t', model.MSE(y_test, predictions))

    # with L2 regularisation
    print('\nRegression with Quadratic kernel with L2 regularisation:\n')
    K_reg = kernel_regression(alpha = 0.1)
    predictions = K_reg.fit(X_train, y_train, X_test, 'quadratic_kernel')
    print('\nSSE: for Quadratic Kernel\t', model.SSE(y_test, predictions))
    print('MSE for Quadratic Kernel:\t', model.MSE(y_test, predictions), '\n')