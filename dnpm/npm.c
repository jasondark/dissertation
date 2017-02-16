#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <cvode/cvode.h>
#include <cvode/cvode_spbcgs.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_types.h>

/* Change these if you are running insane parameters and Sundials is bailing */
#define ATOL 1e-12
#define RTOL 1e-8

/* r=mu/gamma, b=(2*alpha*beta)/(gamma*mu), n=min size */
typedef struct {
	realtype r, b, n;
} UserData;

/* the RHS of the ODE system. entries 0-2 are (X,Y,W), and entries 3+ are u_i */
int f(realtype t, N_Vector Y, N_Vector Ydot, void *user_data) {
	/* get the problem parameters */
	UserData *problem = (UserData*) user_data;
	realtype r = problem->r,
			 b = problem->b,
			 n = problem->n;

	/* Check for recoverable errors. If non-negative, restructure to a more natural notation */
	size_t len = NV_LENGTH_S(Y), i;
	realtype *ptr, *f, *dx, *dy, *dw, *df, x, y, w, fsum;
	ptr = NV_DATA_S(Y);
	for (size_t i = 0; i < len; i++) {
		if (ptr[i] < -ATOL)
			return 1;
	}
	len -= 3;
	x  = ptr[0]; y = ptr[1]; w = ptr[2]; f = ptr+3;
	ptr = NV_DATA_S(Ydot);
	dx = ptr+0; dy = ptr+1; dw = ptr+2; df = ptr+3;



	/* the moment-closed system */
	*dx = r*(1.0-x) + n*(n-1.0)*y - b*x*y;
	*dy = (w-(n-1.0+r))*y;
	*dw = b*x - w*(w+1.0); 

	/* the size density system */
	fsum = 1.0-f[0];
	df[0] = -(b*x + w)*f[0] + 2.0*fsum;
	for (i = 1; i < len; i++) {
		fsum -= f[i];
		/* recoverable error: just decrease step-size */
		if (fsum < -ATOL)
			return 1;

		df[i] = -(b*x + ((realtype) i + w))*f[i] + b*x*f[i-1] + 2.0*fsum;
	}
	return 0;
}



// Usage ./npm r b n x0 y0 w0 imax t1 [t2 t3 ...]
int main(int argc, char** argv) {
	if (argc < 9) {
		printf("Usage: %s r b n x0 y0 w0 imax t1 [t2 [t3 [...]]]\n", argv[0]);
		printf("Output: for each time step (including t=0), a row of the form\nt\tx\ty\tw\tu1\tu2\t...\tuimax\n");
		return 0;
	}
	size_t i, j;

	/* Parse the user input */
	UserData problem;
	problem.r = atof(argv[1]);
	problem.b = atof(argv[2]);
	problem.n = atof(argv[3]);
	realtype x0 = atof(argv[4]);
	realtype y0 = atof(argv[5]);
	realtype w0 = atof(argv[6]);
	size_t dim = 3 + atoi(argv[7]);

	/* This is mostly boiler-plate for a bit */
	realtype* f0   = (realtype*) malloc(dim * sizeof(realtype));
	realtype* atol = (realtype*) malloc(dim * sizeof(realtype));
	N_Vector y, ytol;
	realtype reltol, t, tret;
	void *cvode_mem;
	int flag;
	for (unsigned int i = 0; i < dim; i++) {
		atol[i] = ATOL;
		f0[i] = 0.0;
	}
	f0[0] = x0;
	f0[1] = y0;
	f0[2] = w0;
	f0[3 + (size_t) w0] = 1.0;
	t = 0.0;
	y = N_VMake_Serial(dim, f0);
	ytol = N_VMake_Serial(dim, atol);
	reltol = RTOL;

	cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);

	flag = CVodeSetUserData(cvode_mem, &problem);        assert(flag >= 0);
	flag = CVodeInit(cvode_mem, f, t, y);                assert(flag >= 0);
	flag = CVodeSVtolerances(cvode_mem, reltol, ytol);   assert(flag >= 0);
	flag = CVSpbcg(cvode_mem, PREC_NONE, 0);             assert(flag >= 0);
	flag = CVodeSetMaxNumSteps(cvode_mem, -1);           assert(flag >= 0);

	realtype* ptr;
	j = 8;
	do {
		ptr = NV_DATA_S(y);
		printf("%g\t%g\t%g\t%g", t, ptr[0], ptr[1], ptr[2]);
		for (i = 3; i < dim; i++)
			printf("\t%g", ptr[i]);
		printf("\n");

		if (j == argc)
			break;

		t = atof(argv[j++]);
		flag = CVode(cvode_mem, t, y, &tret, CV_NORMAL);
		if (flag < 0) break;
	} while (1);

	// Cleanup memory
	N_VDestroy_Serial(y);
	N_VDestroy_Serial(ytol);
	CVodeFree(&cvode_mem);

	free(f0);
	free(atol);
	return 0;
}

