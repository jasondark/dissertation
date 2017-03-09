// mcpersist.cpp
// Description:
//   * Perform a single Monte Carlo simulation of in vivo aggregate persistence
//   * Returns the number of aggregates present at the time of cell division

#include <iostream>
#include <random>
#include <vector>

using namespace std;

int main(int argc, char** argv) {
	// args = [ m k beta gamma shape scale iterations ]
	if (argc < 8) {
		cout << "Usage: " << argv[0] << " mass minsize beta gamma shape scale iterations" << endl;
		return 0;
	}

	const unsigned int
		mass = atoi(argv[1]),
		k = atoi(argv[2]);

	const double
		beta  = atof(argv[3]),
		gamma = atof(argv[4]),
		shape = atof(argv[5]),
		scale = atof(argv[6]);

	unsigned int it = atoi(argv[7]);
	unsigned int success = 0;

	random_device rd;
	mt19937 entropy(rd());

	// the time until cell division
	gamma_distribution<double> t(shape,scale);

	// the uniform random generator
	uniform_real_distribution<double> u(0.0, 1.0);

	// we will simply store a list of each aggregate present and their size in no particular order
	vector<unsigned int> aggregates;
	aggregates.reserve(mass/k+1);

	while (it-->0) {
		// initialization for each Gillespie simulation

		// The ending simulation time
		double tmax = t(entropy);
		aggregates.clear();
		aggregates.push_back(k); // we start with a single aggregate of minimum size
		unsigned int m = mass-k;

		double propensity;
		unsigned int j;

loop: // yes, I am using a goto control structure in C++. yes, I know that is crazy.

		// first, calculate the total propensity of the current state
		propensity = 2.0*beta*double(m)*double(aggregates.size());
		for (auto& i : aggregates)
			propensity += gamma*double(i-1);

		// second, check to see if the reaction occurs after cell division
		tmax -= -log(u(entropy))/propensity;
		if (tmax < 0.0)
			goto finish;

		// third, figure out which reaction occurred and update the state
		propensity *= u(entropy);
		// provide a fast path for aggregation
		propensity -= 2.0*beta*double(m)*double(aggregates.size());
		if (propensity < 0.0) {
			j = ceil(-propensity / (2.0*beta*double(m))) - 1;
			aggregates[j]++;
			m--;
			goto loop;
		}
		for (auto& i : aggregates) {
			propensity -= gamma*double(i-1);
			if (propensity < 0.0) { // fragmented aggregate
				j = ceil(-propensity / gamma);
				if (j < k && i-j < k) { // full disintegration
					m += i; // restore the mass

					// if the only aggregate, early exit
					if (aggregates.size() == 1) {
						aggregates.pop_back();
						goto finish;
					}
					// otherwise swap out the last element for the current one
					// (note: no-op if already last element)
					i = aggregates.back();
					aggregates.pop_back();
				}
				else if (j >= k && i-j >= k) { // amplification of aggregates
					aggregates.push_back(i-j);
					i = j;

					// arbitrary early termination for "productive" infections
					if (aggregates.size() >= 8)
						goto finish;
				}
				else { // neutral fragmentation
					m += min(j,i-j);
					i = max(j,i-j);
				}
				goto loop;
			}
		}

finish:	
		if (aggregates.size() > 1)
			success++;
	}
	cout << mass << '\t' << k << '\t' << beta << '\t' << gamma << '\t' << shape << '\t' << scale << '\t' << double(success)/atof(argv[7]) << endl;
	return 0;
}
