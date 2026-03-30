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

.PHONY: all clean clean_report cleanall help

all: $(BIN_NO) $(BIN_AUTO) $(BIN_AVX)

help:
	@echo "Makefile aviable options:"
	@echo "  make $(BIN_NO)    Build without auto vectorization"
	@echo "  make $(BIN_AUTO)       Build with auto vectorization"
	@echo "  make all               Build all the executables"
	@echo "  make clean             Delete all the executables"
	@echo "  make clean_report      Delete all the vectorizations reports (.txt)"
	@echo "  make cleanall          Delete all executables and reports"

# Report dir
$(VEC_REPO_DIR):
	mkdir -p $@

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

# Cleaning
clean:
	rm -f $(BIN_NO) $(BIN_AUTO) $(BIN_AVX)

clean_report:
	rm -fr ./$(VEC_REPO_DIR)

cleanall: clean clean_report
