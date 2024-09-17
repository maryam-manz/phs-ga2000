import numpy as np

def quadratic(a, b, c):
    discriminant = b**2 - 4*a*c
    
    # Check if the discriminant is negative, which means complex roots
    if discriminant < 0:
        real_part = -b / (2*a)
        imaginary_part = np.sqrt(-discriminant) / (2*a)
        return (complex(real_part, imaginary_part), complex(real_part, -imaginary_part))
    
   
    sqrt_discriminant = np.sqrt(discriminant)
    
    if b >= 0:
        root1 = (-b - sqrt_discriminant) / (2*a)
        root2 = (2*c) / (-b - sqrt_discriminant)  # Using an alternate form
    else:
        root1 = (-b + sqrt_discriminant) / (2*a)
        root2 = (2*c) / (-b + sqrt_discriminant)  # Using an alternate form
    
    return (root1, root2)




