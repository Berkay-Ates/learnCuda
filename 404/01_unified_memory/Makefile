NVCC := nvcc
NVCC_FLAGS := -O2 -I../../common
TARGET := unified_memory
SRC := unified_memory.cu

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: run clean
