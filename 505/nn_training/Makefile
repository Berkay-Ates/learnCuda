NVCC := nvcc
NVCC_FLAGS := -O2 -I../../common
TARGET := nn_training
SRC := nn_training.cu

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: run clean
