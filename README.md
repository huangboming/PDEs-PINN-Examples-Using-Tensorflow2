# PDEs-PINN-Examples-Using-Tensorflow2

Some examples of using PINN to solve PDEs numerically.

PINN official implementation: https://github.com/maziarraissi/PINNs

SA-PINN official implementation: https://github.com/levimcclenny/SA-PINNs

Update: 

- Add SA-PINN implementation for below examples:
  - Poisson 2D
  - Diffusion equation
  - Burgers equation
- In SA-PINN implementation, I provide "save model" and "load model" function. Also, I add my trained models to the repo.


----

### Poisson 1D example

Poisson 1D with boundary condition:

$$\left\{ 
    \begin{array}{l}
    -u^{''}(x) = f(x) \quad \textrm{for } \, x \in [-1, 1] \newline
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
    -\Delta u(x, y) = f(x, y) \quad \textrm{for } \, (x, y) \in \Omega = [-1, 1] \times [-1, 1] \newline
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
    u_t = u_{xx} - e^{-t}(\sin(\pi x) - \pi^2\sin(\pi x)) \quad x \in [-1, 1], t\in [0, 1] \newline
    u(x, 0) = \sin(\pi x) \newline
    u(-1, t) = u(1, t) = 0
    \end{array}
\right.
$$

The exact solution is: $u(x, t) = e^{-t}\sin(\pi x)$.

PINN solution v.s. the exact solution: 
![pinn solution 2d](Diffusion%20Equation/outputs/pinn_solution_2d.png)
![real solution 2d](Diffusion%20Equation/outputs/real_solution_2d.png)

![pinn solution 3d](Diffusion%20Equation/outputs/pinn_solution_3d.png)
![real solution 3d](Diffusion%20Equation/outputs/real_solution_3d.png)

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

--- 

### Burgers Equation example

I learned how to use PINN to solve burgers equation from [omniscientoctopus' code](https://github.com/omniscientoctopus/Physics-Informed-Neural-Networks/blob/main/TensorFlow/Burgers%20Equation/Burgers_Equation.ipynb). Much of my code is same as omniscientoctopus' code, but I add some comments and my ideas in my code

Here's omniscientoctopus' GitHub repo: https://github.com/omniscientoctopus/Physics-Informed-Neural-Networks

The data I used is also from omniscientoctopus' GitHub repo.

Burgers equation:
$$\left\{
\begin{array}{l}
u_t + uu_x = \nu u_{xx} \quad x \in [-1, 1], t \in [0, 1] \newline
u(x, 0) = u_0(x) \newline
u(-1, t) = u(1, t) = 0
\end{array}
\right.$$

where $u_0(x) = -\sin(\pi x)$, $\nu = \frac{0.01}{\pi}$

PINN results: 
![real and pinn solution](Burgers%20Equation/outputs/real_and_pinn_solution.png)

SA-PINN results:
![real solution](Burgers%20Equation/outputs/exact_solution_3d.png)
![sa-pinn solution](Burgers%20Equation/outputs/sa-pinn_solution_3d.png)
![real v.s sa-pinn solution](Burgers%20Equation/outputs/real_and_sa-pinn_solution.png)


