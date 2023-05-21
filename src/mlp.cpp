#include "mlp.h"

MultilayerPerceptron::MultilayerPerceptron(const std::initializer_list<size_t>& dimensions, const std::initializer_list<ActivationFunction>& act_funcs, LossFunction loss_func) {
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
    for (size_t i = 0; dims_iterator != dimensions.end(); ++dims_iterator, ++act_func_iterator, ++i) {
        _m_linear_layers[i] = LinearLayer(*(dims_iterator - 1), *dims_iterator);
        _m_non_linear_layers[i] = NonLinearLayer(*act_func_iterator);
        _m_delta_linear_layers[i] = DeltaLinearLayer(*(dims_iterator - 1), *dims_iterator);

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

void MultilayerPerceptron::TrainOnOneSample(const Vector& input, const Vector& output) {
    assert(input.size() == _m_input_size);
    assert(output.size() == _m_output_size);

    std::vector<Vector> computed(_m_num_of_layers);

    for (size_t i = 0; i < _m_num_of_layers; ++i) {
        computed[i] = _m_linear_layers[i].Calculate((i == 0 ? input : computed[i - 1]));
        computed[i] = _m_non_linear_layers[i].Calculate(computed[i]);
    }
    
    
    Vector u = _m_loss.GetDerivative(computed[_m_num_of_layers - 1], output);

    for (size_t i = _m_num_of_layers - 1; i-- > 0;) {
        u = _m_non_linear_layers[i].ThrowDerivative(u);
        
        _m_delta_linear_layers[i].Update_dA(u, computed[i]);
        _m_delta_linear_layers[i].Update_db(u);

        u = _m_linear_layers[i].ThrowDerivative(u);
    }
}

void MultilayerPerceptron::UpdateParameters() {
    for (size_t i = 0; i < _m_num_of_layers; ++i) {
        _m_linear_layers[i].UpdateParameters(_m_delta_linear_layers[i]);
        _m_delta_linear_layers[i].Clear();
    }
}

using DataSet = std::vector<Vector>;

void MultilayerPerceptron::MultilayerPerceptron::Train(const DataSet& input, const DataSet& output) {
    for (size_t i = 0; i < input.size(); i += batch_size) {
        size_t r = std::min(i + batch_size, input.size());

        for (size_t j = i; j < r; ++j) {
            TrainOnOneSample(input[j], output[j]);
        }

        UpdateParameters();
    }
}