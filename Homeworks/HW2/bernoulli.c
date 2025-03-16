#include "bernoulli.h"

#include <fann.h>

#include "lib/tests.h"
#include "lib/utils.h"

int write_training_data_ode() {
  FILE* file = fopen("./lib/ode.data", "w");
  if(!file) {
    perror("Error opening file");
    return 1;  // Exit the program or handle the error
  }

  Points2D results = RK4(fP, fQ, 2, 2, -1, 3, 0.0001);

  fprintf(file, "%d %d %d\n", 10001, 1, 1);

  for(size_t i = 0; i < 100001; i++) {
    fprintf(file, "%-7.6f %-7.6f\n", results.x[i], results.y[i]);
  }

  fclose(file);
  return 0;
}

void train_neural_network_ode() {
  printf("Generating training data...\n");
  write_training_data_ode();

  const unsigned int num_layers = 5;
  const unsigned int layer_neurons[num_layers] = {1, 12, 17, 12, 1};
  const float desired_error = 0.000001;
  const unsigned int max_epochs = 500000;
  const unsigned int epochs_between_reports = 1000;

  struct fann* ann = fann_create_standard_array(num_layers, layer_neurons);

  fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
  fann_set_activation_function_output(ann, FANN_LINEAR);

  // Optimizations
  fann_set_learning_rate(ann, 0.1);
  fann_set_learning_momentum(ann, 0.7);
  fann_set_bit_fail_limit(ann, 0.1);

  fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);  // RPROP = Resilient propagation
  // Fine-tune learning rates RPROP
  fann_set_rprop_increase_factor(ann, 0.9);
  fann_set_rprop_decrease_factor(ann, 0.6);
  fann_set_rprop_delta_min(ann, 0.0);
  fann_set_rprop_delta_max(ann, 49.0);

  fann_set_train_error_function(ann, FANN_ERRORFUNC_TANH);

  printf("Training network...\n");

  fann_train_on_file(ann, "./lib/ode.data", max_epochs, epochs_between_reports, desired_error);

  printf("Training complete!\n");

  fann_save(ann, "ode_solver.net");

  fann_destroy(ann);
}

int inference_from_nn_ode() {
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

int bernoulli() {
  printf("Running bernoulli\n");
  FILE* file = fopen("./lib/ode.data", "r");
  if(file) {
    fclose(file);
    return inference_from_nn_ode();
  } else {
    fclose(file);
    train_neural_network_ode();
    return 0;
  }
}