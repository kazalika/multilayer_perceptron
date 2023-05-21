#include "linear_layer.h"

// begin -- Delta Linear Layer

DeltaLinearLayer::DeltaLinearLayer(size_t input_size, size_t output_size) {
    _dA.resize(output_size, Vector(input_size));
    _db.resize(output_size);
}

void DeltaLinearLayer::Update_dA(const Vector& u, const Vector& z) {
    for (size_t i = 0; i < u.size(); ++i) {
        for (size_t j = 0; j < z.size(); ++j) {
            _dA[i][j] += u[i] * z[j];
        }
    }
}

void DeltaLinearLayer::Update_db(const Vector& u) {
    for (size_t i = 0; i < u.size(); ++i) {
        _db[i] += u[i];
    }
}

const Matrix& DeltaLinearLayer::Get_dA() const {
    return _dA;
}

const Vector& DeltaLinearLayer::Get_db() const {
    return _db;
}

void DeltaLinearLayer::Clear() {
    for (size_t i = 0; i < _dA.size(); ++i) {
        for (size_t j = 0; j < _dA[i].size(); ++j) {
            _dA[i][j] = 0;
        }
    }

    for (size_t i = 0; i < _db.size(); ++i) {
        _db[i] = 0;
    }
}

// end -- Delta Linear Layer

// begin -- Linear Layer

LinearLayer::LinearLayer(size_t input_size, size_t output_size) {
    _A.resize(output_size, Vector(input_size));
    _b.resize(output_size);
}

Vector LinearLayer::Calculate(const Vector& x) const {
    assert(x.size() == _A[0].size());
    
    Vector result(_A.size());
    for (size_t i = 0; i < _A.size(); ++i) {
        for (size_t j = 0; j < _A[i].size(); ++j) {
            result[i] += _A[i][j] * x[j];
        }
        result[i] += _b[i];
    }
    return result;
}

Vector LinearLayer::ThrowDerivative(const Vector& u) const {
    assert(u.size() == _A[0].size());

    Vector result(_A.size());
    for (size_t i = 0; i < _A.size(); ++i) {
        for (size_t j = 0; j < _A[i].size(); ++j) {
            result[i] += _A[i][j] * u[j];
        }
    }
    return result;
}

void LinearLayer::UpdateParameters(const DeltaLinearLayer& delta) {
    const Matrix& dA = delta.Get_dA();
    const Vector& db = delta.Get_db();

    for (size_t i = 0; i < _A.size(); ++i) {
        for (size_t j = 0; j < _A[i].size(); ++j) {
            _A[i][j] -= dA[i][j];
        }
    }

    for (size_t i = 0; i < _b.size(); ++i) {
        _b[i] -= db[i];
    }
}

// end -- Linear Layer
