NVCC := nvcc
NVCC_FLAGS := -O2 -I../../common
TARGET := matrix_mul_naive
SRC := matrix_mul_naive.cu

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: run clean
