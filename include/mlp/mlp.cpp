#include "mlp.h"

#include <iostream>

namespace mlp {

MultilayerPerceptron::MultilayerPerceptron(
    const std::initializer_list<ssize_t>& dimensions,
    const std::initializer_list<ActivationFunction>& act_funcs,
    LossFunction loss_func) {
  assert(dimensions.size() > 1);
  assert(dimensions.size() == act_funcs.size() + 1);

  _m_num_of_layers = dimensions.size() - 1;
  _m_loss = loss_func;

  _m_linear_layers.resize(_m_num_of_layers);
  _m_non_linear_layers.resize(_m_num_of_layers);
  _m_delta_linear_layers.resize(_m_num_of_layers);

  auto dims_iterator = dimensions.begin();
  auto act_func_iterator = act_funcs.begin();
  _m_input_size = *dims_iterator;
  ++dims_iterator;
  for (size_t i = 0; dims_iterator != dimensions.end();
       ++dims_iterator, ++act_func_iterator, ++i) {
    _m_linear_layers[i] = LinearLayer(*(dims_iterator - 1), *dims_iterator);
    _m_non_linear_layers[i] = NonLinearLayer(*act_func_iterator);
    _m_delta_linear_layers[i] =
        DeltaLinearLayer(*(dims_iterator - 1), *dims_iterator);

    if (i + 1 == _m_num_of_layers) {
      _m_output_size = *dims_iterator;
    }
  }
}

Vector MultilayerPerceptron::Calculate(const Vector& input) const {
  assert(input.size() == _m_input_size);

  Vector val = input;
  for (size_t i = 0; i < _m_num_of_layers; ++i) {
    val = _m_linear_layers[i].Calculate(val);
    val = _m_non_linear_layers[i].Calculate(val);
  }

  assert(val.size() == _m_output_size);
  return val;
}

void MultilayerPerceptron::TrainOnOneSample(const Vector& input,
                                            const Vector& output) {
  std::vector<Vector> computed(_m_num_of_layers + 1);
  computed[0] = input;

  for (size_t i = 1; i < computed.size(); ++i) {
    // linear = Ax + b
    Vector linear = _m_linear_layers[i - 1].Calculate(computed[i - 1]);
    // computed[i] = \sigma(linear)
    computed[i] = _m_non_linear_layers[i - 1].Calculate(linear);
  }

  Vector u = _m_loss.GetDerivative(computed.back(), output);

  for (size_t i = _m_num_of_layers; i-- > 0;) {
    // x = z_{i-1}
    Vector x = computed[i];

    // linear = Ax + b
    Vector linear = _m_linear_layers[i].Calculate(x);

    // dS = \sigma'(Ax + b)
    Matrix dS = _m_non_linear_layers[i].ThrowDerivative(linear);

    // \sigma'(Ax + b) * u * x.T
    _m_delta_linear_layers[i].Update_dA(dS * u, x);

    // \sigma'(Ax + b) * u
    _m_delta_linear_layers[i].Update_db(dS * u);

    // u_{i - 1} = (u.T * \sigma'(Ax + b) * A).T
    u = _m_linear_layers[i].ThrowDerivative(dS, u);
  }
}

void MultilayerPerceptron::UpdateParameters() {
  for (size_t i = 0; i < _m_num_of_layers; ++i) {
    _m_linear_layers[i].UpdateParameters(_m_delta_linear_layers[i], batch_size);
    _m_delta_linear_layers[i].Clear();
  }
}

using DataSet = std::vector<std::vector<double>>;

Vector to_Vector(const std::vector<double>& v) {
  Vector result(v.size(), 1);
  for (size_t i = 0; i < v.size(); ++i) {
    result[static_cast<ssize_t>(i)] = v[i];
  }
  return result;
}

void MultilayerPerceptron::MultilayerPerceptron::Train(size_t num_of_iterations,
                                                       const DataSet& input,
                                                       const DataSet& output) {
  for (size_t it = 0; it < num_of_iterations; ++it) {
    for (size_t i = 0; i < input.size(); i += batch_size) {
      size_t r = std::min(i + batch_size, input.size());

      // train on batch
      for (size_t j = i; j < r; ++j) {
        TrainOnOneSample(to_Vector(input[j]), to_Vector(output[j]));
      }

      UpdateParameters();
    }
  }
}

template <typename T>
void WriteInStream(std::ostream& out, T x) {
  out.write(reinterpret_cast<char*>(&x), sizeof(x));
}

template <typename T>
void ReadFromStream(std::istream& in, T& x) {
  in.read(reinterpret_cast<char*>(&x), sizeof(x));
}

void MultilayerPerceptron::SaveModel(const std::string& file_path) const {
  std::ofstream out(file_path, std::ios::binary);

  WriteInStream(out, _m_num_of_layers);
  WriteInStream(out, _m_input_size);
  WriteInStream(out, _m_output_size);

  for (size_t i = 0; i < _m_num_of_layers; ++i) {
    WriteLinearLayer(out, _m_linear_layers[i]);
    WriteActivationFunction(out, _m_non_linear_layers[i].GetActivatioFunc());
  }

  WriteLossFunction(out, _m_loss);
}

void MultilayerPerceptron::LoadModel(const std::string& file_path,
                                     const ActivationFunctionsList& act_list,
                                     const LossFunctionsList& los_list) {
  std::ifstream in(file_path, std::ios::binary);

  ReadFromStream(in, _m_num_of_layers);
  ReadFromStream(in, _m_input_size);
  ReadFromStream(in, _m_output_size);

  for (size_t i = 0; i < _m_num_of_layers; ++i) {
    _m_linear_layers.push_back(ReadLinearLayer(in));
    _m_non_linear_layers.emplace_back(ReadActivationFunction(in, act_list));

    ssize_t input_size = _m_linear_layers[i].GetInputSize();
    ssize_t output_size = _m_linear_layers[i].GetOutputSize();
    _m_delta_linear_layers.emplace_back(input_size, output_size);
  }

  ReadLossFunction(in, los_list);
}

}  // namespace mlp
