#include "non_linear_layer.h"

#include <cassert>

namespace mlp {

namespace activation_functions {

double sig_f(double x) {
  return 1.0 / (1.0 + exp(-x));
}

Vector sigmoid(const Vector& x) {
  Vector result = 1.0 / (1.0 + exp(-x.array()));

  return result;
}

Matrix sigmoid_der(const Vector& x) {
  Matrix result =
      (exp(-x.array()) / pow(1.0 + exp(-x.array()), 2)).matrix().asDiagonal();

  return result;
}

Vector relu(const Vector& x) {
  Vector result = x.cwiseMax(0);
  return result;
}

Matrix relu_der(const Vector& x) {
  Matrix result = (x.array() > 0.0).cast<double>().matrix().asDiagonal();
  return result;
}

Vector softmax(const Vector& x) {
  double sum_exp = x.array().exp().sum();
  Vector result = x.array().exp() / sum_exp;
  return result;
}

Matrix softmax_der(const Vector& x) {
  Vector computed = softmax(x);
  Matrix diagonal = computed.asDiagonal();
  Matrix result = diagonal - computed * computed.transpose();
  return result;
}

}  // namespace activation_functions

template <typename T>
void WriteInStream(std::ostream& out, T x) {
  out.write(reinterpret_cast<char*>(&x), sizeof(x));
}

template <typename T>
void ReadFromStream(std::istream& in, T& x) {
  in.read(reinterpret_cast<char*>(&x), sizeof(x));
}

void WriteActivationFunction(std::ostream& out, const ActivationFunction& f) {
  std::string f_name = f.GetName();
  size_t name_size = f_name.size();

  WriteInStream(out, name_size);
  for (auto c : f_name) {
    WriteInStream(out, c);
  }
}

ActivationFunction ReadActivationFunction(std::istream& in,
                                          const ActivationFunctionsList& list) {
  size_t name_size = 0;
  ReadFromStream(in, name_size);

  std::string f_name;
  for (size_t i = 0; i < name_size; ++i) {
    char c;
    ReadFromStream<char>(in, c);
    f_name += c;
  }

  return list.GetByName(f_name);
}

// begin -- Non Linear Layer

Vector NonLinearLayer::Calculate(const Vector& x) const {
  // \sigma(x)
  return _activation_func.Compute(x);
}

Matrix NonLinearLayer::ThrowDerivative(const Vector& w) const {
  // \sigma'(Ax + b)
  return _activation_func.ComputeDerivative(w);
}

// end -- Non Linear Layer

}  // namespace mlp
