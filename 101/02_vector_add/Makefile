NVCC := nvcc
NVCC_FLAGS := -O2 -I../../common
TARGET := vector_add
SRC := vector_add.cu

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: run clean
