
Derivation of the characteristic equations using a second order approximation.  This method is based on the method described by Chaudhry on page 71 of "Applied Hydraulic Transients" (third edition, 2014).

### Load Sympy Modules


```python
from sympy import *
from sympy.matrices import *
import sympy.mpmath
from sympy.utilities.lambdify import lambdify
init_printing()

import numpy as np00
import scipy.linalg as LA

%matplotlib inline
import matplotlib.pyplot as plt
```


```python
qp, qa, qb, g, A, a, hp, ha, hb, f, dt, theta, d = symbols('Q_p Q_a Q_b g A a H_p H_a H_b f s theta d')
```


```python
B = g*A*sin(theta);B
```




$$A g \sin{\left (\theta \right )}$$




```python
R = Rational(1,2)*f/(d*A); R
```




$$\frac{f}{2 A d}$$




```python
eq1 = qp - qb - g*A/a*(hp-hb) + B + Rational(1,2)*R*dt*(qb**2 +qp**2); eq1
```




$$A g \sin{\left (\theta \right )} - \frac{A g}{a} \left(- H_{b} + H_{p}\right) - Q_{b} + Q_{p} + \frac{f s \left(Q_{b}^{2} + Q_{p}^{2}\right)}{4 A d}$$




```python
eq2 = qp - qa + g*A/a*(hp-ha) + B + Rational(1,2)*R*dt*(qa**2 +qp**2); eq2
```




$$A g \sin{\left (\theta \right )} + \frac{A g}{a} \left(- H_{a} + H_{p}\right) - Q_{a} + Q_{p} + \frac{f s \left(Q_{a}^{2} + Q_{p}^{2}\right)}{4 A d}$$




```python
sol = solve([eq1,eq2],(qp,hp))
q_new = sol[0][0]
h_new = sol[0][1]
```

The future flow rate $(Q_p)$


```python
q_new.simplify()
```




$$\frac{1}{2 a f s} \left(- 4 A a d + \sqrt{2} \sqrt{a \left(4 A^{2} H_{a} d f g s - 4 A^{2} H_{b} d f g s + 8 A^{2} a d^{2} - 8 A^{2} a d f g s \sin{\left (\theta \right )} + 4 A Q_{a} a d f s + 4 A Q_{b} a d f s - Q_{a}^{2} a f^{2} s^{2} - Q_{b}^{2} a f^{2} s^{2}\right)}\right)$$



The future head $(H_p)$


```python
h_new.simplify()
```




$$\frac{1}{8 A^{2} d g} \left(4 A^{2} d g \left(H_{a} + H_{b}\right) + 4 A a d \left(Q_{a} - Q_{b}\right) + a f s \left(- Q_{a}^{2} + Q_{b}^{2}\right)\right)$$


