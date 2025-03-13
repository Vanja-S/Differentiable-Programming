CC = clang
CFLAGS = -Wall -Wextra -O2 # Enable warnings and optimizations
TARGET = bernoulli.out # Output executable name

# Source files (main program + all .c files in the lib folder)
SRCS = bernoulli.c $(wildcard lib/*.c) # Include all .c files in lib folder
OBJS = $(SRCS:.c=.o) # Object files

# FANN library paths and linking
FANN_INCLUDE = -I/opt/homebrew/Cellar/fann/2.2.0/include
FANN_LIB = -L/opt/homebrew/Cellar/fann/2.2.0/lib -lfann

# Include the header files from the lib folder and FANN
CFLAGS += -Ilib $(FANN_INCLUDE)

# Default rule to build the executable
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET) $(FANN_LIB)

# Rule to compile .c files into .o files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up compiled files
clean:
	rm -f $(OBJS) $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)