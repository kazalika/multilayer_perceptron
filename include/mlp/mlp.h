#include <initializer_list>

#include "linear_layer.h"
#include "non_linear_layer.h"
#include "loss_func.h"


class MultilayerPerceptron {
public:
    MultilayerPerceptron(const std::initializer_list<size_t>& dimensions, const std::initializer_list<ActivationFunction>& act_funcs, LossFunction loss_func);

    Vector Calculate(const Vector& input) const;

    void TrainOnOneSample(const Vector& input, const Vector& output);

    void UpdateParameters();

    using DataSet = std::vector<Vector>;

    void Train(const DataSet& input, const DataSet& output);

private:
    size_t _m_num_of_layers;
    size_t _m_input_size;
    size_t _m_output_size;
    std::vector<LinearLayer> _m_linear_layers;
    std::vector<DeltaLinearLayer> _m_delta_linear_layers;
    std::vector<NonLinearLayer> _m_non_linear_layers;

    LossFunction _m_loss;

    size_t batch_size = 10;
};
