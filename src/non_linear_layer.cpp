#include "non_linear_layer.h"

// begin -- Non Linear Layer

Vector NonLinearLayer::Calculate(const Vector& x) const {
    Vector result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = _activation_func.Compute(x[i]);
    }
    return result;
}

Vector NonLinearLayer::ThrowDerivative(const Vector& u) const {
    Vector result(u.size());
    for (size_t i = 0; i < u.size(); ++i) {
        result[i] = _activation_func.ComputeDerivative(u[i]);
    }
    return result;
}

// end -- Non Linear Layer
