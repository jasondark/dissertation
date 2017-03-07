#include <iostream>
#include <cstdint>
#include <cmath>
#include <utility>
#include <fstream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>

#include "assemblyop.hpp"
#include "preconditioner.cpp"

using namespace Eigen;
using namespace std;


enum Preconditioner { NONE, DIAGONAL, BACKWARD, FORWARD };


pair<MatrixXd,MatrixXd> getRates(const string& fname) {
	vector<double> rates;
	double r;

	ifstream file(fname, ios::in);
	if (!file.is_open()) {
		cout << "Error: No such file " << fname << endl;
		exit(1);
	}

	while (file >> r)
		rates.push_back(r);
	file.close();

	unsigned int k = (unsigned int) (sqrt(0.25 + (double) rates.size()) - 0.5);

	if (k*(k+1) != rates.size()) {
		cout << "Error: Entries in " << fname << " are not properly formatted." << endl;
	}

	MatrixXd A(k,k);
	MatrixXd B(k,k);

	unsigned int ind = 0, i, j;
	for (i = 0; i < k; i++) {
		for (j = 0; j < i; j++)
			A(i,j) = A(j,i) = rates[ind++];
		A(i,i) = rates[ind++];
	}
	for (i = 0; i < k; i++) {
		for (j = 0; j < i; j++)
			B(i,j) = B(j,i) = rates[ind++];
		B(i,i) = rates[ind++];
	}

	return make_pair(A,B);
}











int main(int argc, char** argv) {
	if (argc < 4) {
		cout << "Usage:\t" << argv[0] << " mass rates moments [rescale [diagnostic [precond]]]" << endl;
		cout << "mass:\tthe total number of units in the assembly system" << endl;
		cout << "rates:\tthe filename containing the rate specification for the assembly system" << endl;
		cout << "moments:\tthe number of uncentered moments to compute" << endl;
		cout << "rescale:\t(optional) scale A(i,j) by this quantity. Default=1.0" << endl;
		cout << "diagnostic:\t(optional) if present, output only the all-monomer value and the number of iterations." << endl;
		cout << "precond:\t(optional) if precond < 0, performs forward SOR(-precond);" << endl << "        \tif precond = 0 (or absent), no preconditioning;" << endl << "        \tif precond > 0, backward SOR(precond);" << endl;
		return 0;
	}


	uintmax_t mass = atoi(argv[1]);

	string file(argv[2]);
	pair<MatrixXd,MatrixXd> pr = getRates(file);

	size_t nmoments = atoi(argv[3]);

	double sigma = 1.0;
	if (argc > 4)
		sigma = atof(argv[4]);
	bool diagnostic = argc > 5;
	Preconditioner precond = NONE;
	double omega = 0.0;
	if (argc > 6) {
		precond = DIAGONAL;
		omega = atof(argv[6]);
		if (omega < 0.0) {
			precond = FORWARD;
			omega *= -1.0;
		}
		else if (omega > 0.0) {
			precond = BACKWARD;
		}
	}


	MatrixXd B = get<1>(pr);
	size_t n0 = B.rows()+1;

	PartitionList<uintmax_t> states(n0-1, mass);

	MatrixXd A = sigma * get<0>(pr);
	AssemblyOp<uintmax_t> Lambda(states, A, B);
	VectorXd b, c = VectorXd::Constant(states.size(), 1.0);

	if (precond == NONE) {
		BiCGSTAB< AssemblyOp<uintmax_t>, IdentityPreconditioner > solver(Lambda);
		for (size_t i = 1; i <= nmoments; i++) {
			b = ((double) i) * c;
			c = solver.solve(b);
			if (diagnostic)
				cout << c(0) << '\t' << solver.iterations() << endl;
			else
				cout << c.transpose() << endl;
		}
	}
	else if (precond == DIAGONAL) {
		BiCGSTAB< AssemblyOp<uintmax_t>, AssemblyDiagonalPreconditioner<uintmax_t> > solver(Lambda);
		for (size_t i = 1; i <= nmoments; i++) {
			b = ((double) i) * c;
			c = solver.solve(b);
			if (diagnostic)
				cout << c(0) << '\t' << solver.iterations() << endl;
			else
				cout << c.transpose() << endl;
		}
	}
	else if (precond == FORWARD) {
		BiCGSTAB< AssemblyOp<uintmax_t>, AssemblyLowerPreconditioner<uintmax_t> > solver(Lambda);
		solver.preconditioner().setRelaxation(omega);
		for (size_t i = 1; i <= nmoments; i++) {
			b = ((double) i) * c;
			c = solver.solve(b);
			if (diagnostic)
				cout << c(0) << '\t' << solver.iterations() << endl;
			else
				cout << c.transpose() << endl;
		}
	}
	else if (precond == BACKWARD) {
		BiCGSTAB< AssemblyOp<uintmax_t>, AssemblyUpperPreconditioner<uintmax_t> > solver(Lambda);
		solver.preconditioner().setRelaxation(omega);
		for (size_t i = 1; i <= nmoments; i++) {
			b = ((double) i) * c;
			c = solver.solve(b);
			if (diagnostic)
				cout << c(0) << '\t' << solver.iterations() << endl;
			else
				cout << c.transpose() << endl;
		}
	}

	return 0;

}
