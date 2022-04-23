import numpy as np
from skqlearn.utils import inner_product_estimation


def kernel_wrapper(
        data_encoding_method: callable,
        calculation_method: str,
) -> callable:
    """Kernel function wrapper through data encoding methods.

    Quantum data encoding can be interpreted as a feature map. These feature
    maps can then define a kernel by using the inner product.

    todo: Elaborate.

    Args:
        data_encoding_method (callable): Data encoding function that defines
            the kernel.
        calculation_method ({'classic', 'quantum'}): The kernel calculation
            method:
            'classic': The inner product is calculated classically.
            'quantum': The inner product is estimated with a quantum circuit.

    Returns:
        callable: Kernel function than can then be provided to classes such as
            sklearn.svm.SVC to be used as a custom kernel.
    """
    if calculation_method == 'classic':
        def kernel_function(x, y):
            x_samples_list = []
            y_samples_list = []
            for i in range(x.shape[0]):
                x_samples_list.append(data_encoding_method(x[i, :]))

            for i in range(y.shape[0]):
                y_samples_list.append(data_encoding_method(y[i, :]))

            x_encoded = np.vstack(x_samples_list)
            y_encoded = np.vstack(y_samples_list)

            return np.dot(x_encoded, y_encoded.T)

        return kernel_function
    elif calculation_method == 'quantum':
        def kernel_function(x, y):
            x_samples_list = []
            y_samples_list = []
            for i in range(x.shape[0]):
                x_samples_list.append(data_encoding_method(x[i, :]))

            for i in range(y.shape[0]):
                y_samples_list.append(data_encoding_method(y[i, :]))

            x_encoded = np.vstack(x_samples_list)
            y_encoded = np.vstack(y_samples_list)

            # Calculation of the gram matrix
            gram = np.zeros([x.shape[0], y.shape[0]])
            for i in range(x.shape[0]):
                for j in range(y.shape[0]):
                    gram[i, j] = inner_product_estimation(x_encoded[i, :],
                                                          y_encoded[j, :])

            return gram

        return kernel_function
    else:
        raise ValueError(f'Invalid calculation method provided. Expected '
                         f'classic or quantum, got {calculation_method} '
                         f'instead.')