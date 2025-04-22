import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, make_jaxpr
import timeit

# --- Define function and its gradients ---

def f(x1, x2):
    """Compute f(x1, x2) = ln(x1) + x1 * x2 - sin(x2)"""
    return jnp.log(x1) + x1 * x2 - jnp.sin(x2)

# Reverse-mode gradients
dy_dx1 = grad(f, argnums=0)
dy_dx2 = grad(f, argnums=1)

# --- Evaluation at sample point ---

x1, x2 = 2.0, 5.0

print("=== Function and Gradients at (x1=2.0, x2=5.0) ===")
print("f(x1, x2)  =", f(x1, x2))
print("∂f/∂x1     =", dy_dx1(x1, x2))
print("∂f/∂x2     =", dy_dx2(x1, x2))

# --- Print JAXPR for gradient functions ---

print("\n=== JAXPR for ∂f/∂x1 ===")
print(make_jaxpr(dy_dx1)(x1, x2))

print("\n=== JAXPR for ∂f/∂x2 ===")
print(make_jaxpr(dy_dx2)(x1, x2))

# --- JIT compilation and HLO analysis ---

@jit
def jitted_f(x1, x2):
    return f(x1, x2), dy_dx1(x1, x2), dy_dx2(x1, x2)

x1s = jnp.array([2.0])
x2s = jnp.array([5.0])

compiled = jax.xla_computation(jitted_f)(x1s, x2s)
print("\n=== HLO IR (JAX to XLA) ===")
print(compiled.as_hlo_text())

# --- Compare JIT strategies (g1 vs g2) ---

g1 = lambda x1, x2: (jit(f)(x1, x2), jit(dy_dx1)(x1, x2), jit(dy_dx2)(x1, x2))
g2 = jit(lambda x1, x2: (f(x1, x2), dy_dx1(x1, x2), dy_dx2(x1, x2)))

print("\n=== JIT Timing ===")
print("g1:")
print(timeit.timeit(lambda: g1(2.0, 5.0), number=1000), "sec for 1000 runs")

print("g2:")
print(timeit.timeit(lambda: g2(2.0, 5.0), number=1000), "sec for 1000 runs")

# --- Vectorized batch execution using vmap ---

x1s = jnp.linspace(1.0, 10.0, 1000)
x2s = x1s + 1

batch1 = vmap(g2, in_axes=(0, 0))      # batch both x1s and x2s
batch2 = vmap(g2, in_axes=(0, None))   # batch only x1s, fix x2

out1 = batch1(x1s, x2s)
out2 = batch2(x1s, 0.5)

print("\n=== Batched Output Preview (f(x1, x2)) ===")
print("batch1 (x1s, x2s):", out1[0][:5])
print("batch2 (x1s, x2=0.5):", out2[0][:5])