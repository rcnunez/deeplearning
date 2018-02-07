from sympy import *
import sys,re,time

# Initialize Variables
current_x_val = 5       # Start of algorithm
gamma         = 0.01    # Learning rate
precision     = 0.00001
step_count    = 0
previous_step_size = current_x_val

# Parse Arguments
if len(sys.argv) == 5:
	Third = sys.argv[1]
	Secnd = sys.argv[2]
	First = sys.argv[3]
	Const = sys.argv[4]
else:
	print ("Input Error: at least 4 integers separated by spaces are required\n")

# Define
x = Symbol ('x')
f = Function ('f')(x)

# Convert String "expression" to a function to be able to compute the derivative
expression = (int(Third) * x**3) + (int(Secnd) * x**2) + (int(First) * x) + int(Const) 
df = lambdify(x,diff(expression,x),"numpy")

# Loop until derivative is as close to 0 as possible factoring the learning rate
# Based on: https://en.wikipedia.org/wiki/Gradient_descent
while previous_step_size > precision:
    previous_x_val = current_x_val
    current_x_val += -gamma * df(previous_x_val)
    previous_step_size = abs(current_x_val - previous_x_val)
    step_count = step_count + 1

print ("It took %d steps to find the local minimum" % step_count)
print ("The local minimum occurs at %f" % current_x_val)