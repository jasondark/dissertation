SHELL:=/bin/bash

EIGEN:=/usr/local/include


CPPFLAGS := -O3 -std=c++11 $(CPPFLAGS) -I$(EIGEN) -I src -Wno-deprecated-declarations -DNDEBUG

solve: src/solve.cpp src/partition.hpp src/partitionlist.hpp src/assemblyop.hpp src/preconditioner.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) src/solve.cpp -o bin/solve

sweep:
	@echo 'logscale,time,none,forward,backward' && paste -d, <(seq -3 .5 2) \
	<(seq -3 .5 2 | awk '{print 10^$$1}' | xargs -I{} ./bin/solve.exe 100 ./models/cf5.txt 1 {} 1 | awk 'BEGIN{OFS=","} {print $$1,$$2}') \
	<(seq -3 .5 2 | awk '{print 10^$$1}' | xargs -I{} ./bin/solve.exe 100 ./models/cf5.txt 1 {} 1 -1| awk '{print $$2}') \
	<(seq -3 .5 2 | awk '{print 10^$$1}' | xargs -I{} ./bin/solve.exe 100 ./models/cf5.txt 1 {} 1 1 | awk '{print $$2}')

