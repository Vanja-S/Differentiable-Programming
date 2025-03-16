#include <stdio.h>
#include <string.h>

#include "bernoulli.h"
#include "laplace.h"

int main(int argc, char const **args) {
  if(argc < 2) {
    printf("Too few arguments, expected a command which NN to run: bernoulli or laplace!");
    return -1;
  }
  if(strncmp(args[1], "bernoulli", 10) == 0) {
    bernoulli();
  } else if(strncmp(args[1], "laplace", 8) == 0) {
    laplace();
  }
  return 0;
}