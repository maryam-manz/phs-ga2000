import numpy as np
# # Question 5

print("Please enter the value for a")
a = float(input())
print("Please enter the value for b")
b = float(input())
print("Please enter the value for c")
c = float(input())


x1 = (-b+np.sqrt(b**2-4*a*c))/(2*a)
x2 = (-b-np.sqrt(b**2-4*a*c))/(2*a)
print(f'The two roots are : x1 = {x1} and x2 = {x2}')




x3 = 2*c/(-b-np.sqrt(b**2-4*a*c))
x4 = 2*c/(-b+np.sqrt(b**2-4*a*c))

print(f'The two roots are : x3 = {x3} and x4 = {x4}')


# #The problem is that when subtracting numbers, the precision of floating-point representation is reduced because the computer can only store a finite number of digits (typically 16). Subtraction between close numbers can lead to significant accuracy loss



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




root1, root2 = quadratic_roots(0.001, 1000, 0.001)
print(f"The roots of the quadratic equation are: {root1} and {root2}")
