import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.style.use('seaborn-whitegrid')
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 25})
plt.gcf().set_size_inches([8.0, 6.8])

x = np.arange(-3, 4, 1)

def hinge(x):
    if x >= 0:
        return x
    else:
        return 0

y = [hinge(xi) for xi in x]

plt.plot(x, y, '-b', linewidth=3)
plt.xticks(x)
plt.plot(1, 1, 'ro', markersize=7)
plt.annotate('$\\theta^* = [0, 1]$', (1.25, 1))
plt.plot(-1, 0, 'gD', markersize=7)
plt.annotate('$\\theta^* = [1, 0]$', (-1.5, 0.25))
plt.xlabel('y')
plt.ylabel('Objective Value  $\\max\{0, y\}$')
filename = 'plot/'+'hinge.pdf'
plt.savefig(filename, format='pdf')
plt.show()
