FC ?= gfortran
FFLAGS ?= -O2 -mtune=native -march=native
BLAS ?= -lblas

persist: persist.f95
	$(FC) persist.f95 $(BLAS) $(FFLAGS) -o persist


