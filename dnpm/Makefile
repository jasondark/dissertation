# Specify sane defaults for compilation if not otherwise defined
CC     ?= gcc
CFLAGS ?= -Ofast -march=native -mtune=native

# Where is Sundials installed? (may be left blank if in system-default paths)
SUNDIALS ?= -I/usr/local/include -L/usr/local/lib



npm: npm.c
	$(CC) $(CFLAGS) npm.c $(SUNDIALS) -lsundials_cvode -lsundials_nvecserial -o npm

test: npm
	./npm 1 1 5 0.99 0.002 0 10 1

