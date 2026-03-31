CXX      = g++
CXXFLAGS = -std=c++17 -pedantic-errors -Wall
INCLUDES = -I./include
OPTFLAGS = -O3

# Vectorization report
VEC_REPO_DIR=vect_rep
VEC_REPORT_AUTO = -fopt-info-vec-all=./$(VEC_REPO_DIR)/report_$(BIN_AUTO)_info_vec.all
VEC_REPORT_AVX = -fopt-info-vec-all=./$(VEC_REPO_DIR)/report_$(BIN_AVX)_info_vec.all

# Flags
NO_AUTOVEC = -march=native -fno-tree-vectorize
AUTOVEC    = -march=native -ffast-math $(VEC_REPORT_AUTO)
AVX = -march=native -mavx2 -ffast-math $(VEC_REPORT_AVX)

# Executable names
BIN_NO   = bsl_no_autovec
BIN_AUTO = bsl_autovec
BIN_AVX  = bsl_avx
BIN_CUDA = bsl_cuda

.PHONY: all clean clean_report cleanall help

all: $(BIN_NO) $(BIN_AUTO) $(BIN_AVX) $(BIN_CUDA)

help:
	@echo "Makefile aviable options:"
	@echo "  make $(BIN_NO)    Build without auto vectorization"
	@echo "  make $(BIN_AUTO)       Build with auto vectorization"
	@echo "  make $(BIN_AVX)        Build with manual AVX"
	@echo "  make $(BIN_CUDA)          Build with cuda"
	@echo "  make all               Build all the executables"
	@echo "  make clean             Delete all the executables"
	@echo "  make clean_report      Delete all the vectorizations reports (.txt)"
	@echo "  make cleanall          Delete all executables and reports"
	@echo "  make exec_all          Gater data from all the executables"
	@echo "  make exec_$(BIN_NO)    Gater data from the baseline"
	@echo "  make exec_$(BIN_AUTO)    Gater data from the auto vectorized executable"
	@echo "  make exec_$(BIN_AVX)    Gater data from the manual avx executable"
	@echo "  make exec_$(BIN_CUDA)    Gater data from the cuda executable"

# Report dir
$(VEC_REPO_DIR):
	mkdir -p $@

exec_$(BIN_NO): $(BIN_NO)
	./$(BIN_NO) > $(BIN_NO).log 2>&1

exec_$(BIN_AUTO): $(BIN_AUTO)
	./$(BIN_AUTO) > $(BIN_AUTO).log 2>&1

exec_$(BIN_AVX): $(BIN_AVX)
	./$(BIN_AVX) > $(BIN_AVX).log 2>&1

exec_$(BIN_CUDA): $(BIN_CUDA)
	./$(BIN_CUDA) > $(BIN_CUDA).log 2>&1

exec_all: $(BIN_NO) $(BIN_AUTO) $(BIN_AVX) $(BIN_CUDA)
	exec_$(BIN_NO)
	exec_$(BIN_AUTO)
	exec_$(BIN_AVX)
	exec_$(BIN_CUDA)

# Compilation no autovec
$(BIN_NO): pmk.cpp | $(VEC_REPO_DIR)
	mkdir -p $(VEC_REPO_DIR)
	$(CXX) $(CXXFLAGS) -DBASELINE $(INCLUDES) $(NO_AUTOVEC) $< -o $@

# Compilation with autovec
$(BIN_AUTO): pmk.cpp | $(VEC_REPO_DIR)
	mkdir -p $(VEC_REPO_DIR)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $(INCLUDES) $(AUTOVEC) $< -o $@

# Compilation with manual avx
$(BIN_AVX): pmk.cpp | $(VEC_REPO_DIR)
	mkdir -p $(VEC_REPO_DIR)
	$(CXX) $(CXXFLAGS) -DAVX $(OPTFLAGS) $(INCLUDES) $(AVX) $< -o $@

# Compilation with cuda
$(BIN_CUDA): pmk.cpp | $(VEC_REPO_DIR)
	mkdir -p $(VEC_REPO_DIR)
	$(CXX) -c $(CXXFLAGS) -DCUDA $(OPTFLAGS) $< -o $@.o
	nvcc -c -std=c++17 $(OPTFLAGS) kernel.cu -o kernel.o
	nvcc $@.o kernel.o -o $@
	rm $@.o kernel.o

# Cleaning
clean:
	rm -f $(BIN_NO) $(BIN_AUTO) $(BIN_AVX) $(BIN_CUDA)
	rm -f *.o

clean_report:
	rm -fr ./$(VEC_REPO_DIR)

cleanall: clean clean_report
