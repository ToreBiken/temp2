import numpy as np
from layers import Flattener

def check_gradient(f, x, delta=1e-5, tol=1e-4):
    """
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using a two-point formula.
    """
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float64

    fx, analytic_grad = f(x)
    analytic_grad = analytic_grad.copy()
    assert analytic_grad.shape == x.shape

    x_flat = x.ravel()
    analytic_grad_flat = analytic_grad.ravel()
    it = np.nditer(x_flat, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        analytic_grad_at_ix = analytic_grad_flat[ix]

        x_plus_delta = x_flat.copy()
        x_plus_delta[ix] += delta
        fx_plus_delta, _ = f(x_plus_delta.reshape(x.shape))

        x_minus_delta = x_flat.copy()
        x_minus_delta[ix] -= delta
        fx_minus_delta, _ = f(x_minus_delta.reshape(x.shape))

        numeric_grad_at_ix = (fx_plus_delta - fx_minus_delta) / (2 * delta)

        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, atol=tol):
            print(f"Gradients differ at {ix}. Analytic: {analytic_grad_at_ix}, Numeric: {numeric_grad_at_ix}")
            return False

        it.iternext()

    print("Gradient check passed!")
    return True



def check_layer_gradient(layer, x, delta=1e-5, tol=1e-4):
    def helper_func(x):
        output = layer.forward(x)
        output_weight = np.ones_like(output)
        loss = np.sum(output * output_weight)
        d_out = np.ones_like(output) * output_weight
        grad = layer.backward(d_out, x)  # Pass `x` as well
        return loss, grad

    return check_gradient(helper_func, x, delta, tol)







def check_layer_param_gradient(layer, x, param_name, delta=1e-5, tol=1e-4):
    """
    Checks gradient correctness for the parameter of the layer

    Arguments:
      layer: neural network layer, with forward and backward functions
      x: starting point for layer input
      param_name: name of the parameter
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Returns:
      bool indicating whether gradients match or not
    """
    param = layer.params()[param_name]
    initial_w = param.value

    output = layer.forward(x)
    output_weight = np.random.randn(*output.shape)

    def helper_func(w):
        param.value = w
        output = layer.forward(x)
        loss = np.sum(output * output_weight)
        d_out = np.ones_like(output) * output_weight
        layer.backward(d_out)
        grad = param.grad
        return loss, grad

    return check_gradient(helper_func, initial_w, delta, tol)


def check_model_gradient(model, X, y, delta=1e-5, tol=1e-4):
    """
    Checks gradient correctness for all model parameters.

    Args:
        model: The model object containing parameters to check.
        X: Input data.
        y: Labels.
        delta: Small perturbation for numerical gradient estimation.
        tol: Tolerance for gradient comparison.

    Returns:
        bool indicating whether gradients match or not.
    """
    params = model.params()  # Get model parameters
    for param_key in params:
        print(f"Checking gradient for {param_key}")
        param = params[param_key]  # Get the parameter object
        initial_w = param.value.copy()  # Save the initial parameter value

        # Initialize the gradient to be checked
        grad = np.zeros_like(param.value)

        # Compute numerical gradient
        for idx in np.ndindex(param.value.shape):  # Use param.value.shape
            original_value = param.value[idx]

            # Increase by delta
            param.value[idx] = original_value + delta
            loss_plus = model.compute_loss_and_gradients(X, y)

            # Decrease by delta
            param.value[idx] = original_value - delta
            loss_minus = model.compute_loss_and_gradients(X, y)

            # Restore original value
            param.value[idx] = original_value

            # Numerical gradient approximation
            grad[idx] = (loss_plus - loss_minus) / (2 * delta)

        # Get the analytical gradient
        analytical_grad = model.gradients[param_key]  # Change to access gradients

        # Compare the numerical and analytical gradients
        difference = np.linalg.norm(grad - analytical_grad) / (np.linalg.norm(grad) + np.linalg.norm(analytical_grad) + 1e-8)
        if difference > tol:
            print(f"Gradient check failed for {param_key}: {difference:.4g}")
            return False
        else:
            print(f"Gradient check passed for {param_key}.")

    return True
