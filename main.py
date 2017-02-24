import numpy as np


####################################################################################################
# Exercise 1: Interpolation

def lagrange_interpolation(x: np.ndarray, y: np.ndarray) -> (np.poly1d, list):
    """
    Generate Lagrange interpolation polynomial.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    polynomial: polynomial as np.poly1d object
    base_functions: list of base polynomials
    """

    assert (x.size == y.size)

    polynomial = np.poly1d([0])  # Zero for sum
    base_functions = []

    # Generate Lagrange base polynomials and interpolation polynomial
    for i in range(0, x.size):
        term = np.poly1d([1])
        for j in range(0, x.size):
            if j != i:
                term *= np.poly1d([1, -x[j]]) / (x[i] - x[j])

        base_functions.append(term)
        polynomial += term*y[i]

    return polynomial, base_functions




def hermite_cubic_interpolation(x: np.ndarray, y: np.ndarray, yp: np.ndarray) -> list:
    """
    Compute hermite cubic interpolation spline

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points
    yp: derivative values of interpolation points

    Returns:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size == yp.size)

    spline = []

    # compute piecewise interpolating cubic polynomials
    for k in range(0, x.size-1):
        # solve linear system for the coefficients of the spline
        A = np.zeros((4, 4))
        A[0, :] = x[k]**3, x[k]**2, x[k], 1
        A[1, :] = x[k+1]**3, x[k+1]**2, x[k+1], 1
        A[2, :] = 3 * x[k]**2, 2 * x[k], 1, 0
        A[3, :] = 3 * x[k+1]**2, 2 * x[k+1], 1, 0

        b = [y[k], y[k+1], yp[k], yp[k+1]]

        solution = np.linalg.solve(A, b)

        # extract local interpolation coefficients from solution
        spline.append(np.poly1d(solution))

    return spline



####################################################################################################
# Exercise 2: Animation

def natural_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Intepolate the given function using a spline with natural boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)

    # construct linear system with natural boundary conditions
    A = np.zeros(((y.size-1) * 4, (x.size-1) * 4))

    for i in range(0, x.size-1):
        A[4*i, 4*i] = x[i]**3
        A[4*i, 4*i+1] = x[i]**2
        A[4*i, 4*i+2] = x[i]
        A[4*i, 4*i+3] = 1

        A[4*i+1, 4*i] = x[i+1]**3
        A[4*i+1, 4*i+1] = x[i+1]**2
        A[4*i+1, 4*i+2] = x[i+1]
        A[4*i+1, 4*i+3] = 1

    for i in range(0, x.size-2):
        A[4*i+2, 4*i] = 3 * (x[i+1]**2)
        A[4*i+2, 4*i+1] = 2 * x[i+1]
        A[4*i+2, 4*i+2] = 1

        A[4*i+2, 4*i+4] = -3 * (x[i+1]**2)
        A[4*i+2, 4*i+5] = -2 * x[i+1]
        A[4*i+2, 4*i+6] = -1

        A[4*i+3, 4*i] = 6 * x[i+1]
        A[4*i+3, 4*i+1] = 2

        A[4*i+3, 4*i+4] = -6 * x[i+1]
        A[4*i+3, 4*i+5] = -2

    A[-2, 0] = 6 * x[0]
    A[-2, 1] = 2

    A[-1, -4] = 6 * x[x.size-1]
    A[-1, -3] = 2

    b = np.zeros(((y.size-1) * 4))
    for i in range(0, y.size-1):
        b[i*4] = y[i]
        b[i*4+1] = y[i+1]

    # solve linear system for the coefficients of the spline
    solution = np.linalg.solve(A, b)

    # extract local interpolation coefficients from solution
    spline = []
    for i in range(0, x.size-1):
        spline.append(np.poly1d(solution[i*4:i*4+4]))

    return spline


def periodic_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Interpolate the given function with a cubic spline and periodic boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)

    # construct linear system with periodic boundary conditions
    A = np.zeros(((y.size-1) * 4, (x.size-1) * 4))

    for i in range(0, x.size-1):
        A[4*i, 4*i] = x[i]**3
        A[4*i, 4*i+1] = x[i]**2
        A[4*i, 4*i+2] = x[i]
        A[4*i, 4*i+3] = 1

        A[4*i+1, 4*i] = x[i+1]**3
        A[4*i+1, 4*i+1] = x[i+1]**2
        A[4*i+1, 4*i+2] = x[i+1]
        A[4*i+1, 4*i+3] = 1

    for i in range(0, x.size-2):
        A[4*i+2, 4*i] = 3 * (x[i+1]**2)
        A[4*i+2, 4*i+1] = 2 * x[i+1]
        A[4*i+2, 4*i+2] = 1

        A[4*i+2, 4*i+4] = -3 * (x[i+1]**2)
        A[4*i+2, 4*i+5] = -2 * x[i+1]
        A[4*i+2, 4*i+6] = -1

        A[4*i+3, 4*i] = 6 * x[i+1]
        A[4*i+3, 4*i+1] = 2

        A[4*i+3, 4*i+4] = -6 * x[i+1]
        A[4*i+3, 4*i+5] = -2

    A[-2, 0] = 3 * (x[0] ** 2)
    A[-2, 1] = 2 * x[0]
    A[-2, 2] = 1

    A[-1, 0] = 6 * x[0]
    A[-1, 1] = 2

    A[-2, -4] = -3 * (x[x.size-1] ** 2)
    A[-2, -3] = -2 * x[x.size-1]
    A[-2, -2] = -1

    A[-1, -4] = -6 * x[x.size-1]
    A[-1, -3] = -2

    b = np.zeros(((y.size-1) * 4))
    for i in range(0, y.size-1):
        b[i*4] = y[i]
        b[i*4+1] = y[i+1]

    # solve linear system for the coefficients of the spline
    result = np.linalg.solve(A, b)

    # extract local interpolation coefficients from solution
    spline = []
    for i in range(0, x.size-1):
        spline.append(np.poly1d(result[i*4:i*4+4]))

    return spline


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
