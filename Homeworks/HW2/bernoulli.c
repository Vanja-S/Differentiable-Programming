#include <fann.h>

#include "lib/tests.h"
#include "lib/utils.h"

int write_training_data() {
  FILE* file = fopen("./lib/ode.data", "w");
  if(!file) {
    perror("Error opening file");
    return 1;  // Exit the program or handle the error
  }

  Points2D results = RK4(fP, fQ, 2, 2, -1, 3, 0.0001);

  fprintf(file, "%d %d %d\n", 10001, 1, 1);

  for(size_t i = 0; i < 10001; i++) {
    fprintf(file, "%-7.6f %-7.6f\n", results.x[i], results.y[i]);
  }

  fclose(file);
  return 0;
}

void train_neural_network() {
  printf("Generating training data...\n");
  write_training_data();

  const unsigned int num_input = 1;
  const unsigned int num_output = 1;
  const unsigned int num_layers = 5;
  const unsigned int num_neurons_hidden1 = 10;
  const unsigned int num_neurons_hidden2 = 10;
  const unsigned int num_neurons_hidden3 = 8;
  const float desired_error = 0.000001;
  const unsigned int max_epochs = 500000;
  const unsigned int epochs_between_reports = 1000;

  struct fann* ann = fann_create_standard(num_layers, num_input, num_neurons_hidden1,
                                          num_neurons_hidden2, num_neurons_hidden3, num_output);

  fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
  fann_set_activation_function_output(ann, FANN_LINEAR);

  // Optimizations (After adding a third hidden layer, the NN converged (with given error) in 5274
  // epochs)
  fann_set_learning_rate(ann, 0.1);
  fann_set_learning_momentum(ann, 0.8);
  fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);  // RPROP = Resilient propagation
  fann_set_train_error_function(ann, FANN_ERRORFUNC_TANH);

  printf("Training network...\n");

  fann_train_on_file(ann, "./lib/ode.data", max_epochs, epochs_between_reports, desired_error);

  printf("Training complete!\n");

  fann_save(ann, "ode_solver.net");

  fann_destroy(ann);
}

int inference_from_nn() {
  struct fann* ann = fann_create_from_file("ode_solver.net");
  if(!ann) {
    printf("Error: Could not load the neural network\n");
    return 1;
  }

  double x_values[] = {3.0, 3.01, 3.02, 3.03, 3.04, 3.05, 3.06};

  printf("Inference results for dy/dx = x^3*y^2 - (4/x)*y^2:\n");
  printf("------------------------------------------------\n");
  printf("   x   |       y       |      RK4 y     \n");
  printf("-------+---------------+----------------\n");

  double rk_results[] = {-0.026382, -0.025849, -0.025330, -0.024824,
                         -0.024332, -0.023851, -0.023383};

  for(int i = 0; i < 7; i++) {
    fann_type input = x_values[i];

    // Inference
    fann_type* output = fann_run(ann, &input);

    // Output the predicted y value
    printf(" %-5.2f | %-13.10f | %-12.10f\n", x_values[i], output[0], rk_results[i]);
  }

  fann_destroy(ann);

  return 0;
}

int main() {
  FILE* file = fopen("./lib/ode.data", "r");
  if(file) {
    fclose(file);
    return inference_from_nn();
  } else {
    fclose(file);
    train_neural_network();
    return 0;
  }
}