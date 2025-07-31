import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def silu(x):
    return x * sigmoid(x)

def glu(x):
    return 0.5 * sigmoid(x)  # A = 0.5, B = x

def swiglu(x):
    return 0.5 * silu(x)     # A = 0.5, B = x

x = np.linspace(-6, 6, 500)
plt.plot(x, relu(x), label='ReLU', linewidth=2)
plt.plot(x, silu(x), label='SiLU (Swish)', linewidth=2)
plt.plot(x, glu(x), label='GLU (0.5 * sigmoid(x))', linestyle='--')
plt.plot(x, swiglu(x), label='SwiGLU (0.5 * SiLU(x))', linestyle='-.')
plt.title("Activation Functions: SiLU vs GLU vs SwiGLU")
plt.xlabel("x")
plt.ylabel("activation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
