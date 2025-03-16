#pragma once
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>  // For sleep function

// Display size
#define DISPLAY_WIDTH 80
#define DISPLAY_HEIGHT 24

// Simulation parameters
#define DISPLAY_DELAY_MS 100

typedef struct {
  double x;
  double y;
  double u;    // x velocity
  double v;    // y velocity
  int active;  // 1 if particle is active, 0 if it left the domain
} Particle;

void initialize_particles(Particle particles[], int num_particles, double x_min, double y_min);
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
                      double y_max);
void draw_particles(int nx,
                    int ny,
                    Particle particles[],
                    int num_particles,
                    double x_min,
                    double y_min,
                    double x_max,
                    double y_max);