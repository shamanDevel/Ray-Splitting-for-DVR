#include <catch.hpp>
#include <cassert>
#include <renderer_math.cuh>

#include <iostream>

static kernel::PolyExpPoly<6, double, 4, double> createPolyExpPoly()
{
	kernel::Polynomial<3, double> absorption{ 0.5, -2.5, 1.3, -0.2 };
	kernel::Polynomial<3, double> color{ 1.5, 0.3, -0.5, 0.8 };

	const auto emission = absorption * color;
	const auto absorptionInt = absorption.integrate();
	const auto transparency = absorptionInt - absorptionInt.evalAtZero();
	const auto pq = kernel::polyExpPoly(emission, transparency);
	return pq;
}

#define CHECK_MESSAGE(cond, msg) do { INFO(msg); CHECK(cond); } while((void)0, 0)
#define REQUIRE_MESSAGE(cond, msg) do { INFO(msg); REQUIRE(cond); } while((void)0, 0)

TEST_CASE("Quadrature", "[Math]")
{
	const auto pep = createPolyExpPoly();
	const double a = 1;
	const double b = 2;
	
	const double epsilon1 = 1e-7; //for convergence
	const double epsilon2 = 1e-4; //for comparison
	const int maxN = 5000;
	
	// check convergence
	double valueRectangle, valueTrapezoid, valueSimpson, valueSimpsonAdaptive, valuePowerSeries;
	{
		double lastValue = 0;
		int N;
		for (N=2; N<maxN; ++N)
		{
			valueRectangle = pep.integrateRectangle(a, b, N);
			if (abs(valueRectangle - lastValue) < epsilon1)
				break;
			lastValue = valueRectangle;
		}
		REQUIRE_MESSAGE(N < maxN, "rectangular rule did not converge");
		std::cout << "rectangular rule converged after " << N << " iterations to a value of " << valueRectangle << std::endl;
	}
	{
		double lastValue = 0;
		int N;
		for (N = 2; N < maxN; ++N)
		{
			valueTrapezoid = pep.integrateTrapezoid(a, b, N);
			if (abs(valueTrapezoid - lastValue) < epsilon1)
				break;
			lastValue = valueTrapezoid;
		}
		REQUIRE_MESSAGE(N < maxN, "trapezoid rule did not converge");
		std::cout << "trapezoid rule converged after " << N << " iterations to a value of " << valueTrapezoid << std::endl;
	}
	{
		double lastValue = 0;
		int N;
		for (N = 2; N < maxN; N+=2)
		{
			valueSimpson = pep.integrateSimpson(a, b, N);
			if (abs(valueSimpson - lastValue) < epsilon1)
				break;
			lastValue = valueSimpson;
		}
		REQUIRE_MESSAGE(N < maxN, "simpson rule did not converge");
		std::cout << "Simpson rule converged after " << N << " iterations to a value of " << valueSimpson << std::endl;
	}
	{
		int count = 0;
		const auto f = [&count, &pep](double x)
		{
			count++;
			return pep(x);
		};
		//struct CountingFunctor
		//{
		//	const kernel::PolyExpPoly<6, double, 4, double> pep_;
		//	mutable int count_ = 0;
		//	double operator()(double x) const { count_++; return pep_(x); }
		//};
		//CountingFunctor f{ pep, 0 };
		double h = (b - a) / 2;
		int N = 0;
		valueSimpsonAdaptive = kernel::Quadrature::adaptiveSimpson
			<decltype(f), double, double>(
			f, a, b, h, epsilon1, N);
		//count = f.count_;
		REQUIRE_MESSAGE(valueSimpsonAdaptive == Approx(valueSimpson).epsilon(epsilon2),
			"adaptive Simpson did not converge");
		std::cout << "Adaptive simpson rule converged with " << count << " function evaluations to a value of " << valueSimpsonAdaptive << std::endl;
	}
	{
		double lastValue = 0;
		int N;
		for (N = 2; N < maxN; ++N)
		{
			valuePowerSeries = pep.integratePowerSeries(a, b, N);
			//std::cout << "Power series: N=" << N << " -> value=" << valuePowerSeries << std::endl;
			if (abs(valuePowerSeries - lastValue) < epsilon1)
				break;
			lastValue = valuePowerSeries;
		}
		REQUIRE_MESSAGE(N < maxN, "power series rule did not converge");
		std::cout << "power series rule converged after " << N << " iterations to a value of " << valuePowerSeries << std::endl;
	}

	CHECK(valueRectangle == Approx(valueTrapezoid).epsilon(epsilon2));
	CHECK(valueRectangle == Approx(valueSimpson).epsilon(epsilon2));
	CHECK(valueRectangle == Approx(valueSimpsonAdaptive).epsilon(epsilon2));
	CHECK(valueRectangle == Approx(valuePowerSeries).epsilon(epsilon2));
	CHECK(valueTrapezoid == Approx(valueSimpson).epsilon(epsilon2));
	CHECK(valueTrapezoid == Approx(valueSimpsonAdaptive).epsilon(epsilon2));
	CHECK(valueTrapezoid == Approx(valuePowerSeries).epsilon(epsilon2));
	CHECK(valueSimpson == Approx(valuePowerSeries).epsilon(epsilon2));
	CHECK(valueSimpson == Approx(valueSimpsonAdaptive).epsilon(epsilon2));
}