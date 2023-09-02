#include <mlp/mlp.h>

#include <inttypes.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

using Image = std::vector<double>;
using Label = int8_t;

enum { MAGIC_NUMBER_IMAGES = 2051, MAGIC_NUMBER_LABELS = 2049 };

int32_t read_int32(std::ifstream& f) {
  int32_t x = 0;
  f.read(reinterpret_cast<char*>(&x), sizeof(x));
  x = __builtin_bswap32(x);
  return x;
}

double read_pixel(std::ifstream& f) {
  uint8_t pixel;
  f.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
  return static_cast<double>(pixel) / 255.0;
}

uint8_t read_label(std::ifstream& f) {
  uint8_t label;
  f.read(reinterpret_cast<char*>(&label), sizeof(label));
  return label;
}

std::vector<Image> readImages(const std::string& file_path) {
  std::ifstream f(file_path, std::ios::binary);

  assert(f);

  int32_t magic_number, number_of_images, number_of_rows, number_of_columns;

  magic_number = read_int32(f);
  assert(magic_number == MAGIC_NUMBER_IMAGES);

  number_of_images = read_int32(f);
  number_of_rows = read_int32(f);
  number_of_columns = read_int32(f);

  int32_t size_of_image = number_of_rows * number_of_columns;

  std::vector<Image> images(number_of_images);

  for (size_t i = 0; i < static_cast<size_t>(number_of_images); ++i) {
    images[i].resize(size_of_image);

    for (size_t j = 0; j < static_cast<size_t>(size_of_image); ++j) {
      images[i][j] = read_pixel(f);
    }
  }

  return images;
}

std::vector<Label> readLabels(const std::string& file_path) {
  std::ifstream f(file_path, std::ios::binary);

  int32_t magic_number, number_of_items;

  magic_number = read_int32(f);
  assert(magic_number == MAGIC_NUMBER_LABELS);

  number_of_items = read_int32(f);

  std::vector<Label> labels(number_of_items);

  for (size_t i = 0; i < static_cast<size_t>(number_of_items); ++i) {
    labels[i] = read_label(f);
  }

  return labels;
}

std::vector<double> LabelToVector(const Label& l) {
  std::vector<double> r(10, 0);
  r[l] = 1.0;
  return r;
}

mlp::DataSet LabelsToDataSet(const std::vector<Label>& labels) {
  mlp::DataSet data_set(labels.size());
  for (size_t i = 0; i < data_set.size(); ++i) {
    data_set[i] = LabelToVector(labels[i]);
  }
  return data_set;
}

bool IsOk(const mlp::MultilayerPerceptron& model, const Image& image,
          const Label& label) {
  mlp::Vector r = model.Calculate(mlp::to_Vector(image));
  size_t chosen = 0;
  for (ssize_t i = 0; i < r.size(); ++i) {
    if (r[i] > r[chosen]) {
      chosen = i;
    }
  }

  return chosen == static_cast<size_t>(label);
}

double GetAccuracy(const mlp::MultilayerPerceptron& model,
                   const std::vector<Image>& images_test_set,
                   const std::vector<Label>& labels_test_set) {
  size_t test_size = images_test_set.size();
  size_t correct_answers = 0;

  for (size_t i = 0; i < test_size; ++i) {
    if (IsOk(model, images_test_set[i], labels_test_set[i])) {
      correct_answers++;
    }
  }

  double accur_rate =
      static_cast<double>(correct_answers) / static_cast<double>(test_size);
  return accur_rate;
}

int main() {
  auto images_training_set = readImages(
      "/home/kazalika/multilayer_perceptron/examples/digits_recognizer/data/"
      "train-images.idx3-ubyte");
  auto labels_training_set = readLabels(
      "/home/kazalika/multilayer_perceptron/examples/digits_recognizer/data/"
      "train-labels.idx1-ubyte");

  mlp::DataSet X_train = images_training_set;
  mlp::DataSet Y_train = LabelsToDataSet(labels_training_set);

  mlp::ActivationFunctionsList act_funcs;
  mlp::LossFunctionsList loss_funcs;

  mlp::ActivationFunction ReLU = act_funcs.GetByName("relu");
  mlp::ActivationFunction Sigmoid = act_funcs.GetByName("sigmoid");
  mlp::ActivationFunction Softmax = act_funcs.GetByName("softmax");

  mlp::LossFunction L = loss_funcs.GetByName("square");

  mlp::MultilayerPerceptron model({28 * 28, 16, 16, 10}, {ReLU, ReLU, Softmax},
                                  L);

  auto images_test_set = readImages(
      "/home/kazalika/multilayer_perceptron/examples/digits_recognizer/data/"
      "t10k-images.idx3-ubyte");
  auto labels_test_set = readLabels(
      "/home/kazalika/multilayer_perceptron/examples/digits_recognizer/data/"
      "t10k-labels.idx1-ubyte");

  std::cout << "Training started" << std::endl;

  model.Train(5, X_train, Y_train);

  std::cout << "Trained!" << std::endl;

  std::cout << "Accuracy before save is "
            << GetAccuracy(model, images_test_set, labels_test_set) * 100 << "%"
            << std::endl;

  model.SaveModel(
      "/home/kazalika/multilayer_perceptron/examples/digits_recognizer/models/"
      "V1");

  std::cout << "Saved model!" << std::endl;

  mlp::MultilayerPerceptron loaded_model;
  loaded_model.LoadModel(
      "/home/kazalika/multilayer_perceptron/examples/digits_recognizer/models/"
      "V1",
      act_funcs, loss_funcs);
  loaded_model.Train(5, X_train, Y_train);

  std::cout << "Accuracy after load and 5 more iterations is "
            << GetAccuracy(loaded_model, images_test_set, labels_test_set) * 100
            << "%" << std::endl;
}
