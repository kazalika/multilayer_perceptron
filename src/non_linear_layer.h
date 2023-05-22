#include <functional>
#include <vector>
#include <stdio.h>

namespace mlp {

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

class ActivationFunction {
public:
    ActivationFunction(const std::function<double(double)>& func, const std::function<double(double)> der) : _activation_function(func), _derivative(der) {
    }

    double Compute(double x) const {
        return _activation_function(x);
    }

    double ComputeDerivative(double x) const {
        return _derivative(x);
    }
private:
    std::function<double(double)> _activation_function;
    std::function<double(double)> _derivative;
};



class NonLinearLayer {
public:

    NonLinearLayer(const ActivationFunction& act_func) : _activation_func(act_func) {
    }

    Vector Calculate(const Vector& x) const;

    Vector ThrowDerivative(const Vector& u) const;

private:
    ActivationFunction _activation_func;
};

} // namespace mlp
