#include "laplace.h"

#include <fann.h>

#include "./lib/tests.h"
#include "./lib/utils.h"
#include "particles.h"

int write_training_data_pde() {
  FILE* file = fopen("./lib/pde.data", "w");
  if(!file) {
    perror("Error opening file");
    return 1;  // Exit the program or handle the error
  }

  int nx = 50;
  int ny = 50;
  double dx = (1.0 - 0.0) / (nx - 1);
  double dy = (1.0 - 0.0) / (ny - 1);

  double u[nx][ny];
  initialize_domain(nx, ny, u, 0.0, 0.0, 1.0, 1.0);
  Jacobi_method(nx, ny, 10000, 1e-5, u);

  fprintf(file, "%d %d %d\n", 2500, 2, 1);

  // Write data
  for(int i = 0; i < nx; i++) {
    double x = 0.0 + i * dx;
    for(int j = 0; j < ny; j++) {
      double y = 0.0 + j * dy;
      fprintf(file, "%-7.6f %-7.6f %-7.6f\n", x, y, u[i][j]);
    }
  }

  fclose(file);
  return 0;
}

void train_neural_network_pde() {
  printf("Generating training data...\n");
  write_training_data_pde();

  const unsigned int num_layers = 7;
  const unsigned int layer_neurons[num_layers] = {2, 32, 41, 52, 41, 24, 1};
  const float desired_error = 0.00001;
  const unsigned int max_epochs = 80000;
  const unsigned int epochs_between_reports = 1000;

  struct fann* ann = fann_create_standard_array(num_layers, layer_neurons);

  fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
  fann_set_activation_function_output(ann, FANN_LINEAR);

  // Optimizations
  fann_set_learning_rate(ann, 0.2);
  fann_set_learning_momentum(ann, 0.7);
  fann_set_bit_fail_limit(ann, 0.1);
  // weight optimizations
  fann_randomize_weights(ann, -0.5, 0.5);

  fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);  // RPROP = Resilient propagation
  // Fine-tune learning rates RPROP
  fann_set_rprop_increase_factor(ann, 0.8);
  fann_set_rprop_decrease_factor(ann, 0.6);
  fann_set_rprop_delta_min(ann, 0.0);
  fann_set_rprop_delta_max(ann, 49.0);

  fann_set_train_error_function(ann, FANN_ERRORFUNC_LINEAR);

  printf("Training network...\n");

  fann_train_on_file(ann, "./lib/pde.data", max_epochs, epochs_between_reports, desired_error);

  printf("Training complete!\n");

  fann_save(ann, "pde_solver.net");

  fann_destroy(ann);
}

int inference_from_nn_pde() {
  struct fann* ann = fann_create_from_file("pde_solver.net");
  if(!ann) {
    printf("Error: Could not load the neural network\n");
    return 1;
  }

  printf("Inference result simulation of 2D flow on domain x ∈ [0.5,1.0], y ∈ [0.5,1.0]:\n");

  double x_min = 0.5;
  double x_max = 1.0;
  double y_min = 0.0;
  double y_max = 0.5;

  int nx = 50;
  int ny = 50;

  double dx = (x_max - x_min) / (nx - 1);
  double dy = (y_max - y_min) / (ny - 1);

  double u[nx][ny];
  double u_velocity[nx][ny], v_velocity[nx][ny];
  initialize_domain(nx, ny, u, x_min, y_min, x_max, y_max);
  fann_type input[2];

  for(int i = 0; i < nx; i++) {
    double x = 1.0 + i * dx;
    for(int j = 0; j < ny; j++) {
      double y = 1.0 + j * dy;
      input[0] = x;
      input[1] = y;

      fann_type* output = fann_run(ann, input);
      double potential_value = (double)output[0];
      u[i][j] = potential_value;
    }
  }

  // for(size_t i = 0; i < 50; i++) {
  //   printf("i=%-2zu ", i);
  //   for(size_t j = 0; j < 50; j++) {
  //     printf("%8.4f ", u[i][j]);
  //   }
  //   printf("\n");
  // }
  // return 0;

  printf("Infering velocity field...\n");
  calculate_velocity_field(nx, ny, u, u_velocity, v_velocity, x_min, y_min, x_max, y_max);

  printf("Initializing particles...\n");
  int num_particles = 90;
  Particle particles[num_particles];
  printf("Starting flow simulation...\n");
  initialize_particles(particles, num_particles, x_min, y_min);
  printf("Starting flow simulation...\n");

  size_t simulation_steps = 150;
  double time_step = 0.01;
  for(size_t step = 0; step < simulation_steps; step++) {
    update_particles(nx, ny, particles, num_particles, u_velocity, v_velocity, time_step, x_min,
                     y_min, x_max, y_max);
    draw_particles(nx, ny, particles, num_particles, x_min, y_min, x_max, y_max);
  }

  fann_destroy(ann);

  return 0;
}

int laplace() {
  printf("Running laplace\n");
  FILE* file = fopen("./lib/pde.data", "r");
  if(file) {
    fclose(file);
    return inference_from_nn_pde();
  } else {
    fclose(file);
    train_neural_network_pde();
    return 0;
  }
  // testJacobiSimulation();
}