import numpy as np
import matplotlib.pyplot as plt

# ISTA Algorithm
def ISTA(A, b, lam, max_iter=1000):
    x = np.zeros(A.shape[1])
    for _ in range(max_iter):
        x = soft_threshold(x + A.T @ (b - A @ x) / np.linalg.norm(A, ord=2)**2, lam)
    return x

def soft_threshold(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

# FISTA Algorithm
def FISTA(A, b, lam, max_iter=1000):
    x = np.zeros(A.shape[1])
    y = x.copy()
    t = 1
    for _ in range(max_iter):
        x_old = x.copy()
        x = soft_threshold(y + A.T @ (b - A @ y) / np.linalg.norm(A, ord=2)**2, lam)
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        y = x + ((t - 1) / t_new) * (x - x_old)
        t = t_new
    return x

# Classical Optimization Methods
def classical_optimization(A, b):
    from scipy.optimize import minimize
    result = minimize(lambda x: 0.5 * np.linalg.norm(b - A @ x)**2, x0=np.zeros(A.shape[1]), method='L-BFGS-B')
    return result.x

# Data Generation
def generate_data(n, m):
    A = np.random.randn(m, n)
    x_true = np.random.randn(n)
    b = A @ x_true + 0.1 * np.random.randn(m)  # Add some noise
    return A, b

# Visualization
def visualize_results(x_true, x_ista, x_fista, x_classical):
    plt.figure(figsize=(10, 6))
    plt.plot(x_true, label='True Signal', linestyle='--')
    plt.plot(x_ista, label='ISTA', linestyle='-')
    plt.plot(x_fista, label='FISTA', linestyle='-')
    plt.plot(x_classical, label='Classical Optimization', linestyle='-')
    plt.legend()
    plt.title('Comparison of Compressed Sensing Methods')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()

# Main Execution
if __name__ == "__main__":
    n = 100
    m = 50
    A, b = generate_data(n, m)

    x_ista = ISTA(A, b, lam=0.1)
    x_fista = FISTA(A, b, lam=0.1)
    x_classical = classical_optimization(A, b)

    visualize_results(np.random.randn(n), x_ista, x_fista, x_classical)