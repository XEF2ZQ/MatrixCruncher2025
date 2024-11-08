import numpy as np
from numba import njit, prange
from numba import config

config.THREADING_LAYER = 'omp'  # Or 'tbb' for Intel Threading Building Blocks

import numba
if int(numba.__version__.split('.')[1]) >= 55:
    numba_random_supported = True
else:
    numba_random_supported = False

# Parameters
num_equations = 10000         # Number of equations to generate
num_variables = 15000            # Number of variables in each equation
coefficients_range = (1_000_000, 100_000_000)  # Range for coefficients
constants_range = (10_000_000, 10_000_000_000) # Range for constants
output_filename = 'large_integer_equations.txt'

if numba_random_supported:
    # Numba supports random in nopython mode
    @njit(parallel=True, fastmath=True, cache=True)
    def generate_large_integer_equations(num_equations, num_variables, coefficients_range, constants_range):
        data = np.empty((num_equations, num_variables + 1), dtype=np.int64)
        for i in prange(num_equations):
            coefficients = np.random.randint(coefficients_range[0], coefficients_range[1], num_variables)
            data[i, :num_variables] = coefficients
            constant = np.random.randint(constants_range[0], constants_range[1])
            data[i, num_variables] = constant
        return data

    # Generate the equations
    data = generate_large_integer_equations(num_equations, num_variables, coefficients_range, constants_range)
else:
    # Generate random numbers using NumPy
    coefficients = np.random.randint(coefficients_range[0], coefficients_range[1], size=(num_equations, num_variables))
    constants = np.random.randint(constants_range[0], constants_range[1], size=num_equations)

    # Assemble equations using Numba
    @njit(parallel=True, fastmath=True, cache=True)
    def assemble_equations(coefficients, constants):
        num_equations = coefficients.shape[0]
        num_variables = coefficients.shape[1]
        data = np.empty((num_equations, num_variables + 1), dtype=np.int64)
        for i in prange(num_equations):
            data[i, :num_variables] = coefficients[i, :]
            data[i, num_variables] = constants[i]
        return data

    # Generate the equations
    data = assemble_equations(coefficients, constants)

# Function to save data to a file
def save_equations_to_file(data, filename):
    np.savetxt(filename, data, fmt='%d', delimiter=' ')

# Save the equations to a file
save_equations_to_file(data, output_filename)

print(f"{num_equations} equations have been written to {output_filename}.")
