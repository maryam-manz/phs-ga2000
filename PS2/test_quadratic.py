import pytest
from quadratic import quadratic

def test_real_roots():
    # Test for real roots
    result = quadratic(1, -3, 2)  # Equation: x^2 - 3x + 2 = 0
    assert result == (2.0, 1.0) or result == (1.0, 2.0), f"Unexpected result: {result}"

def test_complex_roots():
    # Test for complex roots
    result = quadratic(1, 2, 5)  
    assert result == (complex(-1, 2), complex(-1, -2)) or result == (complex(-1, -2), complex(-1, 2)), f"Unexpected result: {result}"

def test_small_discriminant():
    # Test for case with a small discriminant
    result = quadratic(1, 1e8, 1)
    root1, root2 = result
    assert abs(root1 * root2 - 1) < 1e-10, f"Product of roots not close to expected value: {root1 * root2}"

def test_double_root():
    # Test for double root (discriminant = 0)
    result = quadratic(1, -2, 1) 
    assert result == (1.0, 1.0), f"Unexpected result: {result}"


    
    
    
