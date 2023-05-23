#include <stdio.h>
#include <cassert>
#include <initializer_list>
#include <vector>

#include "../src/linear_layer.h"
#include "../src/loss_func.h"
#include "../src/non_linear_layer.h"

namespace mlp {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using DataSet = std::vector<std::vector<double>>;

class MultilayerPerceptron {
 public:
  MultilayerPerceptron() = default;

  MultilayerPerceptron(
      const std::initializer_list<ssize_t>& dimensions,
      const std::initializer_list<ActivationFunction>& act_funcs,
      LossFunction loss_func);

  Vector Calculate(const Vector& input) const;

  void TrainOnOneSample(const Vector& input, const Vector& output);

  void UpdateParameters();

  void Train(size_t num_of_iterations, const DataSet& input,
             const DataSet& output);

  void SaveModel(const std::string& file_path) const;

  void LoadModel(const std::string& file_path,
                 const ActivationFunctionsList& act_list,
                 const LossFunctionsList& los_list);

 private:
  size_t _m_num_of_layers;
  ssize_t _m_input_size;
  ssize_t _m_output_size;
  std::vector<LinearLayer> _m_linear_layers;
  std::vector<DeltaLinearLayer> _m_delta_linear_layers;
  std::vector<NonLinearLayer> _m_non_linear_layers;

  LossFunction _m_loss;

  size_t batch_size = 200;
};

Vector to_Vector(const std::vector<double>& v);

}  // namespace mlp
