#include <catch.hpp>
#include <array>
#include <iostream>
#include <type_traits>
#include <sstream>

#include <renderer_math.cuh>


TEST_CASE("PolyExpPoly-Interpolate", "[Math]")
{
	kernel::Polynomial<3, double> absorption{ 0.5, -2.5, 1.3, -0.2 };
	kernel::Polynomial<3, double> color{ 1.5, 0.3, -0.5, 0.8};

	auto emission = absorption * color;
	auto absorptionInt = absorption.integrate();
	auto transparency = absorptionInt - absorptionInt.evalAtZero();
	auto pq = kernel::polyExpPoly(emission, transparency);

	std::vector<std::pair<double, double>> intervals = {
		{0.2, 1.0}, {0.5, 1.7}, {0.6, 0.9}
	};
	std::vector<int> Nx = { 500, 1000, 2000 };
	int N_gt = 20000;
	for (const auto& interval : intervals)
	{
		double a = interval.first;
		double b = interval.second;
		INFO("eval in [" << a << ", " << b << "]");
		//compute "ground truth"
		double int_gt = pq.integrateTrapezoid(a, b, N_gt);
		for (int N : Nx)
		{
			INFO("N=" << N);
			{
				INFO("rectangle");
				double int_actual = pq.integrateRectangle(a, b, N);
				CHECK(int_actual == Approx(int_gt).epsilon(3e-3));
			}
			{
				INFO("trapezoid");
				double int_actual = pq.integrateTrapezoid(a, b, N);
				CHECK(int_actual == Approx(int_gt).epsilon(1e-3));
			}
		}
	}
}

TEST_CASE("PolyExpPoly-Derivative", "[Math]")
{
	kernel::Polynomial<3, double> absorption{ 0.5, -2.5, 1.3, -0.2 };
	kernel::Polynomial<3, double> color{ 1.5, 0.3, -0.5, 0.8 };

	const auto emission = absorption * color;
	const auto absorptionInt = absorption.integrate();
	const auto transparency = absorptionInt - absorptionInt.evalAtZero();
	const auto pq = kernel::polyExpPoly(emission, transparency);

	const auto pqd1 = pq.d1();
	const auto pqd2 = pq.d2();

	std::default_random_engine rnd(43);
	std::normal_distribution<double> distr(0, 10);
	static const double dx = 1e-4;
	static const double EPSILON = 1e-3;
	static const double MARGIN = 1e-8;
	for (int i=0; i<100; ++i)
	{
		const double x = distr(rnd);

		const double fx = pq(x);
		const double fxL = pq(x - dx);
		const double fxR = pq(x + dx);
		
		const double d1Ana = pqd1(x);
		const double d2Ana = pqd2(x);

		const double d1Num = (fxR - fxL) / (2 * dx);
		const double d2Num = (-2 * fx + fxL + fxR) / (dx*dx);

		INFO("i=" << i << ", x=" << x);
		INFO("fx=" << fx << ", fxL=" << fxL << ", fxR=" << fxR);
		CHECK(d1Ana == Approx(d1Num).epsilon(EPSILON).margin(MARGIN));
		CHECK(d2Ana == Approx(d2Num).epsilon(EPSILON).margin(MARGIN));
	}
}

TEST_CASE("PolyExpPoly-Expand", "[Math]")
{
	kernel::Polynomial<3, double> absorption{ 0.5, -2.5, 1.3, -0.2 };
	kernel::Polynomial<3, double> color{ 1.5, 0.3, -0.5, 0.8 };

	const auto emission = absorption * color;
	const auto absorptionInt = absorption.integrate();
	const auto transparency = absorptionInt - absorptionInt.evalAtZero();
	const auto pq = kernel::polyExpPoly(emission, transparency);

	const auto pqd2 = pq.d2();

	const auto pqd2_expanded = pqd2.expand();
	
	std::default_random_engine rnd(43);
	std::normal_distribution<double> distr(0, 10);
	static const double dx = 1e-4;
	static const double EPSILON = 1e-3;
	static const double MARGIN = 1e-8;
	for (int i = 0; i < 100; ++i)
	{
		const double x = distr(rnd);
		const double f1 = pqd2(x);
		const double f2 = pqd2_expanded(x);
		REQUIRE(f1 == Approx(f2));
	}
}

TEST_CASE("PolyExpPoly-UpperBound-D2", "[Math]")
{
	std::default_random_engine rnd(42);
	std::normal_distribution<double> distr;
	static const int Samples = 150;

	for (int k = 0; k < 20; ++k) {
		kernel::Polynomial<3, double> absorption{ 0.5, -2.5, 1.3, -0.2 };
		kernel::Polynomial<3, double> color{ 1.5, 0.3, -0.5, 0.8 };
		if (k>0)
		{
			for (int i = 0; i <= 3; ++i) absorption[i] = distr(rnd);
			for (int i = 0; i <= 3; ++i) color[i] = distr(rnd);
		}
		INFO("k=" << k);
		INFO("absorption = " << absorption);
		INFO("color = " << color);

		const auto emission = absorption * color;
		const auto absorptionInt = absorption.integrate();
		const auto transparency = absorptionInt - absorptionInt.evalAtZero();
		const auto pq = kernel::polyExpPoly(emission, transparency);

		const auto pqd2 = pq.d2();
		const auto pqd2_expanded = pqd2.expand();

		for (int j = 0; j < 20; ++j)
		{
			double a1 = std::abs(distr(rnd)), b1 = std::abs(distr(rnd));
			double a2 = std::min(a1, b1), b2 = std::max(a1, b1);
			if (j == 0) { a2 = 0.5; b2 = 1.5; }
			double boundExplicitSimple = pqd2.upperBound(a2, b2);
			double boundExpandedSimple = pqd2_expanded.upperBound(a2, b2);
			double boundRecursiveSimple = pq.derivativeBoundsSimple<2>(a2, b2);
			double boundRecursiveBernstein = pq.derivativeBoundsBernstein<2>(a2, b2);
			double boundExpandedBernstein = pqd2_expanded.absBoundBernstein(a2, b2);
			std::stringstream str;
			str << "j=" << j << ", a=" << a2 << ", b=" << b2
				<< " -> simple=" << boundExplicitSimple
				<< ", expand-simple=" << boundExpandedSimple
				<< ", rec-simple=" << boundRecursiveSimple
				<< ", rec-bernstein=" << boundRecursiveBernstein
				<< ", expand-bernstein=" << boundExpandedBernstein;
			INFO(str.str());
			std::cout << str.str() << std::endl;
			double numBound = -FLT_MAX;
			for (int k = 0; k <= Samples; ++k)
			{
				double x = a2 + (k / double(Samples)) * (b2 - a2);
				numBound = std::max(numBound, std::abs(pqd2.eval(x)));
			}
			static const double MARGIN = (1 + 1e-4);
			REQUIRE(numBound <= MARGIN * boundExplicitSimple);
			REQUIRE(numBound <= MARGIN * boundExpandedSimple);
			REQUIRE(numBound <= MARGIN * boundRecursiveSimple);
			CHECK(numBound <= MARGIN * boundRecursiveBernstein);
			CHECK(numBound <= MARGIN * boundExpandedBernstein);
		}
	}
}

