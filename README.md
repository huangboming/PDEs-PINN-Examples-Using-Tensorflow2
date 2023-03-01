# PDEs-PINN-Examples-Using-Tensorflow2

Some examples of using PINN to solve PDEs numerically.

----

### Poisson 1D example

Poisson 1D with boundary condition:

$$\left\{ 
    \begin{array}{l}
    -u^{''}(x) = f(x) \quad \textrm{for } \, x \in (-1, 1) \newline
    u(-1) = u(1) = 0
    \end{array}
    \right.$$


where $f(x) = \pi^2 \sin(\pi x)$.


The exact solution is: $u(x) = \sin(\pi x)$

PINN solution and the exact solution: ![PINN_Poisson_1D](Poisson%201D/results_for_test_set.png)

---

### Poisson 2D example

Poisson 2D with boundary condition:

$$
\left\{ 
    \begin{array}{l}
    -\Delta u(x, y) = f(x, y) \quad \textrm{for } \, (x, y) \in \Omega = (-1, 1) \times (-1, 1) \newline
    u|_{\partial \Omega} = 0
    \end{array}
\right.
$$


where $f(x, y) = 2\pi^2 \sin(\pi x)\sin(\pi y)$.

The exact solution is: $u(x, y) = \sin(\pi x) \sin(\pi y)$

PINN solution: ![PINN_Poisson_2D](Poisson%202D/output/u_pred.png)

The exact solution: ![Exact_Poisson_2D](Poisson%202D/output/u_exact.png)

---

### Diffusion Equation example

(Example from Juan Diego Toscano's [Leaning PIML in Python](https://github.com/jdtoscano94/Learning-Python-Physics-Informed-Machine-Learning-PINNs-DeepONets), example 4: diffusion equation)

Diffusion equation with initial condition and boundary conditions:


$$
\left\{ 
    \begin{array}{l}
    u_t = u_xx - e^{-t}(\sin(\pi x) - \pi^2\sin(\pi x)) \quad x \in [-1, 1], t\in [0, 1] \newline
    u(x, 0) = \sin(\pi x) \newline
    u(-1, t) = u(1, t) = 0
    \end{array}
\right.
$$

The exact solution is: $u(x, t) = e^{-t}\sin(\pi x)$.

PINN solution: 

The exact solution:

---

### Poisson 1D (Inverse Problem) example

Poisson 1D (Inverse Problem) with boundary condition:

PDE:
$$\left\{\begin{array}{l}
-\lambda u^{''}(x) = f(x) \quad x \in (-1, 1) \\
u(-1) = u(1) = 0
\end{array}\right.$$

where $f(x) = \pi^2 \sin(\pi x)$.

Exact solution(for generating training set) is: $u(x) = \sin(\pi x)$.

We're going to train a DNN to approximate $\lambda$. (The exact result is $\lambda = 1$)

PINN solution:

![Poisson 1D results](Poisson%201D%20Inverse%20Problem/model_results.png)
![Poisson 1D lambda history](Poisson%201D%20Inverse%20Problem/lambda_progress.png)




