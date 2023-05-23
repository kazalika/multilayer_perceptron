#include <Eigen/Core>
#include <Eigen/Dense>
#include <EigenRand/EigenRand>

#include <Eigen/src/Core/Matrix.h>
#include <stdio.h>
#include <cassert>
#include <vector>

namespace mlp {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

class DeltaLinearLayer {
 public:
  DeltaLinearLayer() = default;

  DeltaLinearLayer(ssize_t input_size, ssize_t output_size);

  void Update_dA(const Vector& u, const Vector& z);

  void Update_db(const Vector& u);

  const Matrix& Get_dA() const;

  const Vector& Get_db() const;

  void Clear();

 private:
  Matrix _dA;
  Vector _db;
};

class LinearLayer {
 public:
  LinearLayer() = default;

  LinearLayer(ssize_t input_size, ssize_t output_size);

  Vector Calculate(const Vector& x) const;

  Vector ThrowDerivative(const Matrix& dS, const Vector& u) const;

  void UpdateParameters(const DeltaLinearLayer& delta, size_t batch_size);

  Matrix& GetARef();

  Vector& GetbRef();

  const Matrix& GetARef() const;

  const Vector& GetbRef() const;

  ssize_t GetInputSize() const;

  ssize_t GetOutputSize() const;

 private:
  Matrix _A;
  Vector _b;
};

void WriteLinearLayer(std::ostream& out, const LinearLayer& layer);

LinearLayer ReadLinearLayer(std::istream& in);

}  // namespace mlp
