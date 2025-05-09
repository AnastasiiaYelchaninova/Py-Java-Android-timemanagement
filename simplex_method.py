import numpy as np

def simplex_method(matrix):
    # convert matrix to standard form
    m, n = matrix.shape
    matrix = np.hstack((matrix, np.eye(m)))
    c = np.zeros(n + m)
    c[:n] = -1  # maximizing the loss function

    # initial basis solution
    basis = np.arange(n, n + m)

    while True:
        # search for the lead column
        entering_idx = np.argmin(c)
        if c[entering_idx] >= 0:
            break  # means that optimal solution is found

        # search for lead line
        ratios = matrix[:, -1] / matrix[:, entering_idx]
        ratios[ratios <= 0] = np.inf
        leaving_idx = np.argmin(ratios)

        basis[leaving_idx] = entering_idx # updating the base

        # recalculating the table
        pivot = matrix[leaving_idx, entering_idx]
        matrix[leaving_idx, :] /= pivot
        for i in range(m):
            if i != leaving_idx:
                ratio = matrix[i, entering_idx]

# Tests data: 3 examples
payment_matrix1 = np.array([[1, 4, 2, 1, 3], [2, 5, 3, 5, 1], [4, 1, 3, 1, 3], [4, 5, 2, 2, 3]])
result1 = simplex_method(payment_matrix1)
print("Optimal solution:")
print(result1)
print()

payment_matrix2 = np.array([[4, 5, 2, 2, 3], [2, 4, 3, 4, 1], [4, 1, 3, 1, 3], [1, 5, 2, 1, 3]])
result2 = simplex_method(payment_matrix2)
print("Optimal solution:")
print(result2)
print()

payment_matrix3 = np.array([[3, 5, 1, 4, 3], [1, 4, 2, 5, 1], [4, 2, 3, 5, 3], [5, 1, 2, 1, 3]])
result3 = simplex_method(payment_matrix3)
print("Optimal solution:")
print(result3)
