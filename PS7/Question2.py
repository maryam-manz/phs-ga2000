import numpy as np
from scipy.optimize import brent

def f(x):
    """The function to minimize: (x - 0.3)^2 * exp(x)."""
    return (x - 0.3) ** 2 * np.exp(x)

def brent_method(f, a, b, tol=1e-10, max_iter=500):
    """Robust implementation of Brent's method with parabolic interpolation and golden-section fallback."""
    phi = (3 - np.sqrt(5)) / 2  # Golden ratio
    
    
    # Initial points
    x1 = a + phi * (b - a)
    x2 = b - phi * (b - a)
    f1, f2 = f(x1), f(x2)

    for _ in range(max_iter):
        if np.abs(b - a) < tol:
            return (a + b) / 2  # Return the midpoint as the minimum

        # Attempt parabolic interpolation if possible
        try:
            # Compute the minimum of the parabola through (a, f(a)), (x1, f1), (b, f(b))
            u = x1 - ((x1 - a) ** 2 * (f1 - f(b)) - (x1 - b) ** 2 * (f1 - f(a))) / (
                2 * ((x1 - a) * (f1 - f(b)) - (x1 - b) * (f1 - f(a)))
            )

             # If interpolation is valid and makes progress, use it
            if a < u < b and np.abs(u - x1) < (b - a) / 64:
                f_u = f(u)  # Evaluate the function at the new point
                if f_u < f1:
                    b, x2, f2 = x2, u, f_u  # Update the interval to [a, u]
                else:
                    a, x1, f1 = u, x1, f1  # Update the interval to [u, b]
                continue  # Skip to the next iteration if interpolation is used
            else:
                # Fall back to golden-section search
                if f1 < f2:
                    b, x2, f2 = x2, x1, f1
                    x1 = a + phi * (b - a)
                    f1 = f(x1)
                else:
                    a, x1, f1 = x1, x2, f2
                    x2 = b - phi * (b - a)
                    f2 = f(x2)
                    
        except ZeroDivisionError:
            # If parabolic interpolation fails, use golden-section search
            if f1 < f2:
                b, x2, f2 = x2, x1, f1
                x1 = a + phi * (b - a)
                f1 = f(x1)
            else:
                a, x1, f1 = x1, x2, f2
                x2 = b - phi * (b - a)
                f2 = f(x2)

    raise ValueError("Brent's method did not converge within the given iterations.")

# Test the function y = (x - 0.3)^2 * exp(x)
def f(x):
    return (x - 0.3) ** 2 * np.exp(x)

# Test with an appropriate interval
a, b = 0, 1  # Search interval around the expected minimum
min_x_custom = brent_method(f, a, b)

print(f"Minimum at x = {min_x_custom:.8f}, f(x) = {f(min_x_custom):.8f}")


# Test the implemented Brent method
a, b = 0, 1  # Define the search interval
min_x_custom = brent_method(f, a, b)

# Use scipy's brent method for comparison
min_x_scipy = brent(f, brack=(a, b))

# Results
results = {
    "Custom Brent's Method": (min_x_custom, f(min_x_custom)),
    "SciPy Brent's Method": (min_x_scipy, f(min_x_scipy)),
}

print(results)
