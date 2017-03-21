// SIR_VexCL.cpp : Defines the entry point for the console application.
//

#include <iostream>
#include <fstream>
#include <math.h>
#include <vexcl/vexcl.hpp>

#include <boost/numeric/odeint.hpp>
//[ vexcl_includes
#include <boost/numeric/odeint/external/vexcl/vexcl.hpp>



typedef vex::multivector< double, 201 >    result_vector_type;
typedef vex::multivector< double, 3> state_type;

const double beta = 0.5;
const double gama = 1.0 / 3.0;
const double dt = 0.01;


const double t_max = 200.0;

namespace odeint = boost::numeric::odeint;

using namespace std;
using namespace odeint;

struct sys_func
{

	sys_func() { }

	void operator()(const state_type &x, state_type &dxdt, const double &t) const
	{
		//dS = -beta*S*I/N
		dxdt(0) = -beta * (x(0) * x(1)) / (x(0) + x(1) + x(2));

		//I = +beta*S*I/N - gama*I;
		dxdt(1) = beta * (x(0) * x(1)) / (x(0) + x(1) + x(2)) - gama*x(1);

		//R = gama*I
		dxdt(2) = gama*x(1);
	}
};

void output(const std::vector< double > & rS, const std::string &filename,const size_t & n) {

	ofstream myfile;
	myfile.open(filename);

	for (size_t j = 0; j < 201; j++) {
		for (size_t i = 0; i < n; ++i)
		{
			myfile << rS[j * n + i] << "\t";
		}
		myfile << endl;
	}
	myfile.close();
}


struct obs_func
{
	result_vector_type &RS;
	result_vector_type &RI;
	result_vector_type &RR;

	obs_func(result_vector_type &_RS, result_vector_type &_RI, result_vector_type &_RR) : RS(_RS), RI(_RI), RR(_RR) { }

	void operator()(const state_type &x, const double &t)
	{
		if (((int)(t / dt) % 100 )== 0)
		{
			int i = (int)(t / dt) / 100;
			RS(i) = x(0);
			RI(i) = x(1);
			RR(i) = x(2);
		}
		
	}
};

int main(int argc, char** argv)
{
	// set up number of system, time step and integration time
	size_t n = 5;
	//cout << argv[0] << "\t" << argv[1] << endl;
	if (argc > 1)
	{
		n = atoi(argv[1]);
	}



	// setup the opencl context
	vex::Context ctx(vex::Filter::GPU && vex::Filter::DoublePrecision);
	//vex::Context ctx(vex::Filter::DoublePrecision);
	std::cout << ctx << std::endl;


	// initialize R

	result_vector_type RS(ctx.queue(), n);
	result_vector_type RI(ctx.queue(), n);
	result_vector_type RR(ctx.queue(), n);

	// initialize the state of the lorenz ensemble
	state_type X(ctx.queue(), n);
	X(0) = 10000.0;
	X(1) = 1.0;
	X(2) = 0.0;

	// create a stepper
	/*runge_kutta4<
		state_type, double, state_type, double,
		odeint::vector_space_algebra, odeint::default_operations
	> stepper;*/
	
	runge_kutta4<
		state_type
	> stepper;

	// solve the system
	integrate_const(stepper, sys_func(), X, 0.0, t_max, dt, obs_func(RS, RI, RR));


	std::vector< double > rS(201 * n);
	vex::copy(RS, rS);
	std::vector< double > rI(201 * n);
	vex::copy(RI, rI);

	std::vector< double > rR(201 * n);
	vex::copy(RR, rR);

	//output(rS, "rS.txt",n);
	//output(rI, "rI.txt",n);
	//output(rR, "rR.txt",n);


	return 0;
}

