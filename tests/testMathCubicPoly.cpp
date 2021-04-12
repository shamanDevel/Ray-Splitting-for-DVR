#include <catch.hpp>
#include <array>
#include <iostream>
#include <type_traits>

#include <renderer_math.cuh>

TEST_CASE("Cubic-Roots fixed", "[Math][!hide]")
{
	//ax ^ 3 + bx ^ 2 + cx +
	//float4(a = x, b = y, c = z, d = w)

	struct TestCase
	{
		double4 factors;
		int numRoots;
		double roots[3];
	};
	std::vector<TestCase> cases = {
		{{0,0,0,1}, 0, {0,0,0}}, //constant
		{{0,0,2,0}, 1, {0,0,0}}, //linear
		{{0,0,2,-1}, 1, {0.5,0,0}},
		{{0, 1, 0, -1}, 2, {-1, 1, 0}}, //quadratic
		{{0, 1,1,-2}, 2, {-2, 1, 0}},
		{{0, 2, 12, 18}, 1, {-3,0,0}},
		{{0, -0.5, 1.5, 5}, 2, {-2, 5, 0}},
		{{1, -6, -9, 54}, 3, {-3, 3, 6}}, //cubic
		{{-0.25, 2.475, 0.25, 0}, 3, {-0.1, 0, 10}},
		{{10, -100, -0.1, 1}, 3, {-0.1, 0.1, 10}},
		{{-2, -2, 66, -126}, 3, {-7, 3, 3}},
		{{-0.011625, 0.272012, 0.715664, -0.038286}, 3, {-2.4754, 0.04478, 25.3732}}
	};
	double tMin = -20;
	double tMax = +20;
	
	SECTION("Hyperbolic")
	{
		for (const auto& testCase : cases)
		{
			INFO("f(x) = " << testCase.factors.x << " x^3 + " <<
				testCase.factors.y << " x^2 + " << testCase.factors.z << " x + " <<
				testCase.factors.w)
			{
				double roots[3];
				int numRoots = kernel::CubicPolynomial<double>::rootsHyperbolic(testCase.factors, roots);
				CHECK(numRoots == testCase.numRoots);
				std::sort(roots, roots + numRoots);
				for (int i = 0; i < min(numRoots, testCase.numRoots); ++i)
					CHECK(roots[i] == Approx(testCase.roots[i]).margin(1e-3));
			}
		}
	}
	SECTION("Schwarze")
	{
		for (const auto& testCase : cases)
		{
			INFO("f(x) = " << testCase.factors.x << " x^3 + " <<
				testCase.factors.y << " x^2 + " << testCase.factors.z << " x + " <<
				testCase.factors.w)
			{
				double roots[3];
				int numRoots = kernel::CubicPolynomial<double>::rootsSchwarze(testCase.factors, roots);
				REQUIRE(numRoots == testCase.numRoots);
				std::sort(roots, roots + numRoots);
				for (int i = 0; i < min(numRoots, testCase.numRoots); ++i)
					REQUIRE(roots[i] == Approx(testCase.roots[i]).margin(1e-3));
			}
		}
	}
}

TEST_CASE("Cubic-Roots random", "[Math][!hide]")
{
	std::default_random_engine rnd(42);
	std::uniform_int_distribution<int> distrI(0, 1);
	std::normal_distribution<double> distr1(0, 1);
	std::normal_distribution<double> distr2(0, 10);
	const int NUM_TESTS = 10;
	for (int i=0; i<NUM_TESTS; ++i)
	{
		double4 factors;
		factors.x = distrI(rnd) ? distr2(rnd) : distr1(rnd);
		factors.y = distrI(rnd) ? distr2(rnd) : distr1(rnd);
		factors.z = distrI(rnd) ? distr2(rnd) : distr1(rnd);
		factors.w = distrI(rnd) ? distr2(rnd) : distr1(rnd);
		INFO("f(x) = " << factors.x << "*x^3 + " <<
			factors.y << "*x^2 + " << factors.z << "*x + " <<
			factors.w)
		{
			double rootsHyperbolic[3];
			double rootsSchwarze[3];
			int numHyperbolic = kernel::CubicPolynomial<double>::rootsHyperbolic(factors, rootsHyperbolic);
			int numSchwarze = kernel::CubicPolynomial<double>::rootsSchwarze(factors, rootsSchwarze);
			CHECK(numHyperbolic == numSchwarze);
			std::sort(rootsHyperbolic, rootsHyperbolic + numHyperbolic);
			std::sort(rootsSchwarze, rootsSchwarze + numSchwarze);
			for (int i = 0; i < min(numHyperbolic, numSchwarze); ++i) {
				INFO("rootsHyperbolic[i]=" << rootsHyperbolic[i] << ", rootsSchwarze[i]=" << rootsSchwarze[i]);
				CHECK(rootsHyperbolic[i] == Approx(rootsSchwarze[i]).margin(1e-5));
				CHECK(kernel::CubicPolynomial<double>::evalCubic(factors, rootsHyperbolic[i]) == Approx(0.0).margin(1e-3));
				CHECK(kernel::CubicPolynomial<double>::evalCubic(factors, rootsSchwarze[i]) == Approx(0.0).margin(1e-3));
			}
		}
	}
}

