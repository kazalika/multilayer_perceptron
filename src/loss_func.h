#include <stdio.h>
#include <vector>
#include <cassert>
#include <functional>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

class LossFunction {
public:
    double CalculateLoss(const Vector& x, const Vector& y) const {
        assert(x.size() == y.size());

        double result = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            result += _loss(x[i], y[i]);
        }
        return result;
    }

    Vector GetDerivative(const Vector& x, const Vector& y) const {
        assert(x.size() == y.size());

        Vector u(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            u[i] = _loss_derivative(x[i], y[i]);
        }
        return u;
    }

private:
    std::function<double(double, double)> _loss;
    std::function<double(double, double)> _loss_derivative;
};