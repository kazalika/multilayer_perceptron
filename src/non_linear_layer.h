#include <Eigen/Core>
#include <Eigen/Dense>

#include <math.h>
#include <stdio.h>
#include <cmath>
#include <functional>
#include <ostream>
#include <string_view>
#include <vector>

namespace mlp {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

namespace activation_functions {

Vector sigmoid(const Vector& x);
Matrix sigmoid_der(const Vector& x);

Vector relu(const Vector& x);
Matrix relu_der(const Vector& x);

Vector softmax(const Vector& x);
Matrix softmax_der(const Vector& x);

}  // namespace activation_functions

using AFunction = std::function<Vector(const Vector&)>;
using ADerivative = std::function<Matrix(const Vector&)>;

class ActivationFunction {
 public:
  ActivationFunction()
      : _activation_function(activation_functions::sigmoid),
        _derivative(activation_functions::sigmoid_der) {}

  ActivationFunction(const AFunction& func, const ADerivative& der,
                     const std::string& name)
      : _activation_function(func), _derivative(der), _function_name(name) {}

  Vector Compute(const Vector& x) const { return _activation_function(x); }

  Matrix ComputeDerivative(const Vector& x) const { return _derivative(x); }

  std::string GetName() const { return _function_name; }

 private:
  AFunction _activation_function;
  ADerivative _derivative;
  std::string _function_name;
};

class ActivationFunctionsList {
 public:
  ActivationFunctionsList() {
    _functions_list = {
        {activation_functions::sigmoid, activation_functions::sigmoid_der,
         "sigmoid"},
        {activation_functions::relu, activation_functions::relu_der, "relu"},
        {activation_functions::softmax, activation_functions::softmax_der,
         "softmax"},
    };
  }

  void Clear() {
    _functions_list = {
        {activation_functions::sigmoid, activation_functions::sigmoid_der,
         "sigmoid"},
        {activation_functions::relu, activation_functions::relu_der, "relu"},
        {activation_functions::softmax, activation_functions::softmax_der,
         "softmax"},
    };
  }

  void InsertFunction(const AFunction& func, const ADerivative& der,
                      const std::string& name) {
    _functions_list.emplace_back(func, der, name);
  }

  ActivationFunction GetByName(const std::string& name) const {
    for (const auto& f : _functions_list) {
      if (f.GetName() == name) {
        return f;
      }
    }

    // didn't found function :(  => return sigmoid
    return _functions_list[0];
  }

 private:
  std::vector<ActivationFunction> _functions_list;
};

void WriteActivationFunction(std::ostream& out, const ActivationFunction& f);

ActivationFunction ReadActivationFunction(std::istream& in,
                                          const ActivationFunctionsList& list);

class NonLinearLayer {
 public:
  NonLinearLayer() = default;

  NonLinearLayer(const ActivationFunction& act_func)
      : _activation_func(act_func) {}

  Vector Calculate(const Vector& x) const;

  Matrix ThrowDerivative(const Vector& w) const;

  ActivationFunction GetActivatioFunc() const { return _activation_func; }

 private:
  ActivationFunction _activation_func;
};

}  // namespace mlp
