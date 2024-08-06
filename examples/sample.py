import sympy as sp

# Define symbols
x1, x2, u = sp.symbols('x1 x2 u')
theta1, theta2 = sp.symbols('theta1 theta2')
Gamma = sp.symbols('Gamma')

# State vector and parameter vector
x = sp.Matrix([x1, x2])
theta = sp.Matrix([theta1, theta2])
theta_hat = sp.Matrix([sp.symbols('theta1_hat theta2_hat')])

# Define the dynamics of the double integrator
f = sp.Matrix([x2, theta1 * x1])
g = sp.Matrix([0, theta2])

# Define the barrier function
h = x1 + x2 - 1  # Example barrier function

# Compute Lie derivatives
L_f_h = sp.diff(h, x1) * f[0] + sp.diff(h, x2) * f[1]
L_g_h = sp.diff(h, x1) * g[0] + sp.diff(h, x2) * g[1]

# Adaptive Control Barrier Function condition
lambda_cbf = theta_hat - Gamma * sp.Matrix([sp.diff(h, theta1), sp.diff(h, theta2)]).T
acbf_condition = sp.diff(h, x1) * (f[0] + g[0] * lambda_cbf[0]) + sp.diff(h, x2) * (f[1] + g[1] * lambda_cbf[1]) + sp.diff(h, x1) * g[0] * u + sp.diff(h, x2) * g[1] * u

# Update law for parameter estimates
tau = -sp.Matrix([sp.diff(h, x1), sp.diff(h, x2)]).T * f
theta_hat_dot = Gamma * tau

# Display the results
print("Barrier Function h(x):")
sp.pprint(h)

print("\nLie Derivative L_f h(x):")
sp.pprint(L_f_h)

print("\nLie Derivative L_g h(x):")
sp.pprint(L_g_h)

print("\nAdaptive CBF Condition:")
sp.pprint(acbf_condition)

print("\nParameter Update Law:")
sp.pprint(theta_hat_dot)

