#include "linear_layer.h"

#include <iostream>

namespace mlp {

// begin -- Delta Linear Layer

DeltaLinearLayer::DeltaLinearLayer(ssize_t input_size, ssize_t output_size) {
  _dA = Matrix::Zero(output_size, input_size);
  _db = Vector::Zero(output_size);
}

void DeltaLinearLayer::Update_dA(const Vector& u, const Vector& z) {
  assert(u.rows() == _dA.rows());
  assert(z.rows() == _dA.cols());

  // u = \sigma'(Az + b) * u
  // dA = \sigma'(Az + b) * u * z

  _dA += u * z.transpose();
}

void DeltaLinearLayer::Update_db(const Vector& u) {
  // u = \sigma'(Az + b)
  _db += u;
}

const Matrix& DeltaLinearLayer::Get_dA() const {
  return _dA;
}

const Vector& DeltaLinearLayer::Get_db() const {
  return _db;
}

void DeltaLinearLayer::Clear() {
  _dA.setZero();
  _db.setZero();
}

// end -- Delta Linear Layer

// begin -- Linear Layer

LinearLayer::LinearLayer(ssize_t input_size, ssize_t output_size) {
  _A = Matrix::Random(output_size, input_size);
  _b = Vector::Random(output_size);
}

Vector LinearLayer::Calculate(const Vector& x) const {
  if (x.rows() != _A.cols()) {
    std::cout << x.rows() << " vs " << _A.cols() << std::endl;
  }

  assert(x.rows() == _A.cols());
  assert(_A.rows() == _b.rows());

  return _A * x + _b;
}

Vector LinearLayer::ThrowDerivative(const Matrix& dS, const Vector& u) const {
  assert(u.rows() == _A.rows());
  assert(u.rows() == dS.rows());
  assert(u.rows() == dS.cols());

  // u * dS(Ax + b) * A
  Vector result = (u.transpose() * dS * _A).transpose();

  return result;
}

void LinearLayer::UpdateParameters(const DeltaLinearLayer& delta,
                                   size_t batch_size) {
  const Matrix& dA = delta.Get_dA();
  const Vector& db = delta.Get_db();

  _A -= dA / static_cast<double>(batch_size);
  _b -= db / static_cast<double>(batch_size);
}

Matrix& LinearLayer::GetARef() {
  return _A;
}

Vector& LinearLayer::GetbRef() {
  return _b;
}

const Matrix& LinearLayer::GetARef() const {
  return _A;
}

const Vector& LinearLayer::GetbRef() const {
  return _b;
}

ssize_t LinearLayer::GetInputSize() const {
  return _A.cols();
}

ssize_t LinearLayer::GetOutputSize() const {
  return _b.size();
}

// end -- Linear Layer

// begin -- save and read functions

template <typename T>
void WriteInStream(std::ostream& out, T x) {
  out.write(reinterpret_cast<char*>(&x), sizeof(x));
}

template <typename T>
void ReadFromStream(std::istream& in, T& x) {
  in.read(reinterpret_cast<char*>(&x), sizeof(x));
}

void WriteLinearLayer(std::ostream& out, const LinearLayer& layer) {
  auto A = layer.GetARef();
  auto b = layer.GetbRef();

  WriteInStream(out, A.rows());
  WriteInStream(out, A.cols());

  for (ssize_t i = 0; i < A.rows(); ++i) {
    for (ssize_t j = 0; j < A.cols(); ++j) {
      WriteInStream<double>(out, A(i, j));
    }
  }

  WriteInStream(out, b.size());

  for (ssize_t i = 0; i < b.size(); ++i) {
    WriteInStream(out, b[i]);
  }
}

LinearLayer ReadLinearLayer(std::istream& in) {
  ssize_t A_rows, A_cols;
  ReadFromStream(in, A_rows);
  ReadFromStream(in, A_cols);

  LinearLayer layer(A_cols, A_rows);
  auto& A = layer.GetARef();
  auto& b = layer.GetbRef();

  for (ssize_t i = 0; i < A.rows(); ++i) {
    for (ssize_t j = 0; j < A.cols(); ++j) {
      ReadFromStream(in, A(i, j));
    }
  }

  ssize_t b_size;
  ReadFromStream(in, b_size);

  for (ssize_t i = 0; i < b.size(); ++i) {
    ReadFromStream(in, b[i]);
  }

  return layer;
}

// end -- save and read functions

}  // namespace mlp
