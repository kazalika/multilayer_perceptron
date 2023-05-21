#include <vector>
#include <stdio.h>
#include <cassert>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

class DeltaLinearLayer {
public:

    DeltaLinearLayer(size_t input_size, size_t output_size);

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
    
    LinearLayer(size_t input_size, size_t output_size);

    Vector Calculate(const Vector& x) const;

    Vector ThrowDerivative(const Vector& u) const;

    void UpdateParameters(const DeltaLinearLayer& delta);

private:
    Matrix _A;
    Vector _b;
};
