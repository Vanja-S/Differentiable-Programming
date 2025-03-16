#include "particles.h"

void initialize_particles(Particle particles[], int num_particles, double x_min, double y_min) {
  // Initialize particles in a grid pattern
  int rows = (int)sqrt(num_particles);
  int cols = (num_particles + rows - 1) / rows;

  int p = 0;
  for(int i = 0; i < rows && p < num_particles; i++) {
    for(int j = 0; j < cols && p < num_particles; j++) {
      particles[p].x = x_min + 0.1 + 0.8 * j / (double)(cols - 1);
      particles[p].y = y_min + 0.1 + 0.8 * i / (double)(rows - 1);
      particles[p].u = 0.0;
      particles[p].v = 0.0;
      particles[p].active = 1;
      p++;
    }
  }
}

// Function to update particle positions
void update_particles(int nx,
                      int ny,
                      Particle particles[],
                      int num_particles,
                      double u_velocity[nx][ny],
                      double v_velocity[nx][ny],
                      double dt,
                      double x_min,
                      double y_min,
                      double x_max,
                      double y_max) {
  double dx = (x_max - x_min) / (nx - 1);
  double dy = (y_max - y_min) / (ny - 1);

  for(int p = 0; p < num_particles; p++) {
    if(!particles[p].active) continue;

    // Get grid indices for current particle position
    int i = (int)((particles[p].x - x_min) / dx);
    int j = (int)((particles[p].y - y_min) / dy);

    // Ensure within bounds
    if(i < 0 || i >= nx - 1 || j < 0 || j >= ny - 1) {
      particles[p].active = 0;
      continue;
    }

    // Linear interpolation factors
    double x_frac = (particles[p].x - (x_min + i * dx)) / dx;
    double y_frac = (particles[p].y - (y_min + j * dy)) / dy;

    // Bilinear interpolation of velocity
    double u_interp = (1 - x_frac) * (1 - y_frac) * u_velocity[i][j] +
                      x_frac * (1 - y_frac) * u_velocity[i + 1][j] +
                      (1 - x_frac) * y_frac * u_velocity[i][j + 1] +
                      x_frac * y_frac * u_velocity[i + 1][j + 1];

    double v_interp = (1 - x_frac) * (1 - y_frac) * v_velocity[i][j] +
                      x_frac * (1 - y_frac) * v_velocity[i + 1][j] +
                      (1 - x_frac) * y_frac * v_velocity[i][j + 1] +
                      x_frac * y_frac * v_velocity[i + 1][j + 1];

    // Update particle velocities
    particles[p].u = u_interp;
    particles[p].v = v_interp;

    // Update particle positions
    particles[p].x += particles[p].u * dt;
    particles[p].y += particles[p].v * dt;

    // Check if particle is still in domain
    if(particles[p].x < x_min || particles[p].x > x_max || particles[p].y < y_min ||
       particles[p].y > y_max) {
      particles[p].active = 0;
    }
  }
}

void clear_screen() {
  printf("\033[H\033[J");  // ANSI escape code
}

void draw_particles(int nx,
                    int ny,
                    Particle particles[],
                    int num_particles,
                    double x_min,
                    double y_min,
                    double x_max,
                    double y_max) {
  char display[DISPLAY_HEIGHT][DISPLAY_WIDTH + 1];

  for(int j = 0; j < DISPLAY_HEIGHT; j++) {
    memset(display[j], ' ', DISPLAY_WIDTH);
    display[j][DISPLAY_WIDTH] = '\0';
  }

  // Draw domain boundaries
  for(int i = 0; i < DISPLAY_WIDTH; i++) {
    display[0][i] = '-';
    display[DISPLAY_HEIGHT - 1][i] = '-';
  }
  for(int j = 0; j < DISPLAY_HEIGHT; j++) {
    display[j][0] = '|';
    display[j][DISPLAY_WIDTH - 1] = '|';
  }

  // Draw particles
  for(int p = 0; p < num_particles; p++) {
    if(!particles[p].active) continue;

    // Map particle coordinates to display coordinates
    int display_x = (int)((particles[p].x - x_min) / (x_max - x_min) * (DISPLAY_WIDTH - 2)) + 1;
    int display_y =
        (int)((1.0 - (particles[p].y - y_min) / (y_max - y_min)) * (DISPLAY_HEIGHT - 2)) + 1;

    // Ensure within display bounds
    if(display_x >= 1 && display_x < DISPLAY_WIDTH - 1 && display_y >= 1 &&
       display_y < DISPLAY_HEIGHT - 1) {
      // Use different characters based on velocity magnitude
      double vel_mag = sqrt(particles[p].u * particles[p].u + particles[p].v * particles[p].v);
      char particle_char = '.';
      if(vel_mag > 0.5) particle_char = 'o';
      if(vel_mag > 1.0) particle_char = 'O';
      if(vel_mag > 1.5) particle_char = '@';

      display[display_y][display_x] = particle_char;
    }
  }

  clear_screen();
  printf("2D Incompressible Flow Simulation (Laplace Equation)\n");
  for(int j = 0; j < DISPLAY_HEIGHT; j++) {
    printf("%s\n", display[j]);
  }
  printf("Press Ctrl+C to exit\n");

  usleep(DISPLAY_DELAY_MS * 1000);
}
