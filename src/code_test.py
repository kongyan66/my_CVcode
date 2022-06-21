import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, 69, 1)
plt.plot(t, t, 'r', t, t**2, 'b')
label = ['t', 't**2']
plt.legend(label, loc='upper left')
plt.savefig('./test2.jpg')
