#include <fann.h>

#include "lib/tests.h"
#include "lib/utils.h"

void train_neural_network() {
  const unsigned int num_input = 1;
  const unsigned int num_output = 1;
  const unsigned int num_layers = 4;
  const unsigned int num_neurons_hidden1 = 10;
  const unsigned int num_neurons_hidden2 = 6;
  const float desired_error = 0.00001;
  const unsigned int max_epochs = 500000;
  const unsigned int epochs_between_reports = 1000;

  struct fann* ann = fann_create_standard(num_layers, num_input, num_neurons_hidden1,
                                          num_neurons_hidden2, num_output);

  fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
  fann_set_activation_function_output(ann, FANN_LINEAR);

  fann_set_learning_rate(ann, 0.1);

  printf("Training network...\n");

  fann_train_on_file(ann, "./lib/ode.data", max_epochs, epochs_between_reports, desired_error);

  printf("Training complete!\n");

  fann_save(ann, "ode_solver.net");

  fann_destroy(ann);
}

int inference_from_nn(FILE* net_file) {
  struct fann* ann = fann_create_from_file("ode_solver.net");
  if(!ann) {
    printf("Error: Could not load the neural network\n");
    return 1;
  }

  double x_values[] = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5};
  int num_points = 6;

  printf("Inference results for dy/dx = x^2*y^2 - x*y^2:\n");
  printf("------------------------------------------------\n");
  printf("   x   |       y       \n");
  printf("-------+---------------\n");

  for(int i = 0; i < num_points; i++) {
    fann_type input = x_values[i];

    // Inference
    fann_type* output = fann_run(ann, &input);

    // Output the predicted y value
    printf(" %.2f  | %.10f\n", x_values[i], output[0]);
  }

  fann_destroy(ann);

  return 0;
}

int main(int argc, char const** args) {
  FILE* file = fopen("./ode_solver.net", "r");
  if(file) {
    return inference_from_nn(file);
  } else {
    train_neural_network();
  }
  return 0;
}