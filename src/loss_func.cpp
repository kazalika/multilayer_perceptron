#include "loss_func.h"

namespace mlp {

namespace loss_functions {

double square_loss(const Vector& x, const Vector& y) {
  return (x - y).squaredNorm() / static_cast<double>(x.size());
}

Vector square_loss_der(const Vector& x, const Vector& y) {
  return 2 * (x - y);
}

}  // namespace loss_functions

template <typename T>
void WriteInStream(std::ostream& out, T x) {
  out.write(reinterpret_cast<char*>(&x), sizeof(x));
}

template <typename T>
void ReadFromStream(std::istream& in, T& x) {
  in.read(reinterpret_cast<char*>(&x), sizeof(x));
}

void WriteLossFunction(std::ostream& out, const LossFunction& f) {
  std::string f_name = f.GetName();
  size_t name_size = f_name.size();

  WriteInStream(out, name_size);
  for (auto c : f_name) {
    WriteInStream(out, c);
  }
}

LossFunction ReadLossFunction(std::istream& in, const LossFunctionsList& list) {
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

}  // namespace mlp
