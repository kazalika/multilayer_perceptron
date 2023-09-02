#include <Eigen/Core>
#include <Eigen/Dense>

#include <stdio.h>
#include <cassert>
#include <functional>
#include <vector>

namespace mlp {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

using LFunction = std::function<double(const Vector& x, const Vector& y)>;
using LDerivative = std::function<Vector(const Vector& x, const Vector& y)>;

namespace loss_functions {

double square_loss(const Vector& x, const Vector& y);
Vector square_loss_der(const Vector& x, const Vector& y);

}  // namespace loss_functions

class LossFunction {
 public:
  LossFunction()
      : _loss(loss_functions::square_loss),
        _loss_derivative(loss_functions::square_loss_der),
        _name("square") {}

  LossFunction(const LFunction& func, const LDerivative& der,
               const std::string& name)
      : _loss(func), _loss_derivative(der), _name(name) {}

  double CalculateLoss(const Vector& x, const Vector& y) const {
    assert(x.size() == y.size());

    double result = _loss(x, y);
    return result;
  }

  Vector GetDerivative(const Vector& x, const Vector& y) const {
    assert(x.size() == y.size());

    Vector u = _loss_derivative(x, y);
    return u;
  }

  std::string GetName() const { return _name; }

 private:
  std::function<double(const Vector&, const Vector&)> _loss;
  std::function<Vector(const Vector&, const Vector&)> _loss_derivative;
  std::string _name;
};

class LossFunctionsList {
 public:
  LossFunctionsList() {
    _functions_list = {{loss_functions::square_loss,
                        loss_functions::square_loss_der, "square"}};
  }

  void Clear() {
    _functions_list = {{loss_functions::square_loss,
                        loss_functions::square_loss_der, "square"}};
  }

  void InsertFunction(const LFunction& func, const LDerivative& der,
                      const std::string& name) {
    _functions_list.emplace_back(func, der, name);
  }

  LossFunction GetByName(const std::string& name) const {
    for (const auto& f : _functions_list) {
      if (f.GetName() == name) {
        return f;
      }
    }

    // didn't found function :(  => return square
    return _functions_list[0];
  }

 private:
  std::vector<LossFunction> _functions_list;
};

void WriteLossFunction(std::ostream& out, const LossFunction& f);

LossFunction ReadLossFunction(std::istream& in, const LossFunctionsList& list);

}  // namespace mlp
