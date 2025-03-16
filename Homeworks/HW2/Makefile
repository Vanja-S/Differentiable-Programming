CC = clang
CFLAGS = -Wall -Wextra -O2 # Enable warnings and optimizations
TARGET = main.out # Output executable name

# Source files (main program + all .c files in the lib folder)
SRCS = main.c bernoulli.c laplace.c $(wildcard lib/*.c) # Include all .c files including submodules
OBJS = $(SRCS:.c=.o) # Object files

# FANN library paths and linking
FANN_INCLUDE = -I/opt/homebrew/Cellar/fann/2.2.0/include
FANN_LIB = -L/opt/homebrew/Cellar/fann/2.2.0/lib -lfann

# Include the header files from the lib folder and FANN
CFLAGS += -Ilib $(FANN_INCLUDE)

# Default rule to build the executable
all: $(TARGET)

# Rule to build the target executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(FANN_LIB)

# Rule to compile .c files into .o files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up compiled files
clean:
	rm -f $(OBJS) $(TARGET)

# Run the program with a specific submodule
run: $(TARGET)
	@if [ -z "$(filter-out run,$(MAKECMDGOALS))" ]; then \
		echo "Usage: make run <submodule>"; \
		echo "Available submodules: bernoulli, laplace"; \
	else \
		./$(TARGET) $(filter-out $@,$(MAKECMDGOALS)); \
	fi

# Phony targets for the submodules to avoid conflicts with files of the same name
.PHONY: all clean run bernoulli laplace
bernoulli laplace:
	@: # Do nothing, just define the targets so make doesn't look for files with these names