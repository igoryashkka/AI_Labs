import numpy as np
import matplotlib.pyplot as plt

csv_file_path = 'sample006.csv'

data = np.genfromtxt(csv_file_path, delimiter=',')

summator_Z = [0] * len(data)  

w_x = 0
w_y = 0
w_0 = 0

Res_error = 0

active_func = [0] * len(data) 
delta_error = [0] * len(data) 

x = data[:, 0]
y = data[:, 1]
classes = data[:, 2]

counter_epoch = 300

for p in range(counter_epoch):
    for i in range(len(x)):
        # Output of neuron 
        summator_Z[i] = (w_x * x[i]) + (w_y * y[i]) + w_0
    
    for j in range(len(x)):
        if summator_Z[j] > 0:
            active_func[j] = 1
        else:
            active_func[j] = 0

    for i in range(len(classes)):
        # error =       actual class - predicted class
        delta_error[i] = classes[i] - active_func[i]
        
    res = 0
    for j in range(len(x)):
        w_x += x[j] * delta_error[j]
        w_y += y[j] * delta_error[j]
        w_0 += delta_error[j]
        res += np.abs(delta_error[j])
    if p % 10 == 0:
        print(res)
    
    # Plotting the decision boundary
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, c=classes, cmap='viridis', label='Data Points real')
    plt.subplot(1, 2, 2)
    plt.scatter(x, y, c=active_func, cmap='viridis', label='Data Points exp')
    
    # Plotting the decision boundary line
    X = np.linspace(min(x), max(x), 100)
    Y = (-w_0 - w_x * X) / w_y
    plt.plot(X, Y, color='red', linestyle='--')
    plt.ylim([min(y), max(y)])
    
    plt.show()
