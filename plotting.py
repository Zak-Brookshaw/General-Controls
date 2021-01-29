import matplotlib.pyplot as plt


def plotter(X, Y, x_axis='Time', y_axis='Control Var'):
    
    plt.figure()
    plt.plot(X, Y)
    plt.xlabel(x_axis)
    plt.ylabel((y_axis))