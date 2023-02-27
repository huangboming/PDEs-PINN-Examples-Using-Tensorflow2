# PDEs-PINN-Examples-Using-Tensorflow2

Some examples of using PINN to solve PDEs numerically.

----

Poission 1D example: 

$$\left\{ 
    \begin{array}{l}
    -u^{''}(x) = f(x) \quad \textrm{for } \, x \in (-1, 1) \newline
    u(-1) = u(1) = 0
    \end{array}
    \right.$$


where $f(x) = \pi^2 \sin(\pi x)$.


The exact solution is: $u(x) = \sin(\pi x)$

PINN solution and the exact solution: ![PINN_Possion_1D](Possion%201D/results_for_test_set.png)

---

Poission 2D example: 

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

PINN solution: ![PINN_Possion_2D](Possion%202D/output/u_pred.png)

The exact solution: ![Exact_Possion_2D](Possion%202D/output/u_exact.png)