import jax
import jax.numpy as jnp

# Constants for different cases
moon_earth = 0.0123  # Moon/Earth mass ratio
earth_sun = 3.003e-6  # Earth/Sun mass ratio
jupiter_sun = 9.545e-4  # Jupiter/Sun mass ratio

# Initial guess for all cases (0.5 is a reasonable starting point)
r0 = 0.5

def f(r, m_prime):
    """Function representing the rescaled equation."""
    return 1 / r**2 - m_prime / (1 - r)**2 - r

def df_dr(r, m_prime):
    """Analytic derivative of the function."""
    return -2 / r**3 - 2 * m_prime / (1 - r)**3 - 1

def newton_method(f, df, r0, m_prime, tol=1e-10, max_iter=100):
    """Newton's method for finding the root of a function."""
    r = r0
    for i in range(max_iter):
        f_val = f(r, m_prime)
        df_val = df(r, m_prime)
        r_next = r - f_val / df_val

        if jnp.abs(r_next - r) < tol:
            return r_next, i  # Root found
        r = r_next

    raise ValueError("Newton's method did not converge")

# Parameters
m_prime = 0.0123  # Mass ratio of Moon to Earth (approximately)
r0 = 0.5  # Initial guess (between 0 and 1)

# Solve for r'
r_solution, iterations = newton_method(f, df_dr, r0, m_prime)

print(f"Solution for r': {r_solution:.8f} (converged in {iterations} iterations)")
print(f"Distance from Earth to L1: {r_solution * 384400:.2f} km")

# Solve for Moon-Earth L1
r_moon_earth, iter_moon_earth = newton_method(f, df_dr, r0, moon_earth)

# Solve for Earth-Sun L1
r_earth_sun, iter_earth_sun = newton_method(f, df_dr, r0, earth_sun)

# Solve for Jupiter-Sun L1 at Earth's orbit
r_jupiter_sun, iter_jupiter_sun = newton_method(f, df_dr, r0, jupiter_sun)

# Scale distances to km for Earth-Moon and Earth-Sun cases
distance_earth_moon = r_moon_earth * 384400  # km (Earth-Moon distance)
distance_earth_sun = r_earth_sun * 1.496e8  # km (AU distance)
distance_jupiter_sun = r_jupiter_sun * 1.496e8  # km (AU distance)

# Print the results
print(f"Moon-Earth L1: r' = {r_moon_earth:.8f}, Distance = {distance_earth_moon:.2f} km, Iterations = {iter_moon_earth}")
print(f"Earth-Sun L1: r' = {r_earth_sun:.8f}, Distance = {distance_earth_sun:.2f} km, Iterations = {iter_earth_sun}")
print(f"Jupiter-Sun L1: r' = {r_jupiter_sun:.8f}, Distance = {distance_jupiter_sun:.2f} km, Iterations = {iter_jupiter_sun}")
