#include <catch.hpp>
#include <array>
#include <iostream>
#include <type_traits>
#include <random>

#include <renderer_math.cuh>


TEST_CASE("Bernstein", "[Math]")
{
	const kernel::Polynomial<3, double> f{ 0.5, -2.5, 1.3, -0.2 };;
	const auto b = kernel::Bernstein<3, double>::fromPolynomial(f);

	//check for equality
	for (int i=0; i<=100; ++i)
	{
		double x = i / 100.0;
		INFO("i=" << i << ", x=" << x);
		REQUIRE(f(x) == Approx(b(x)));
	}

	//check bounds
	double upperCoarse = f.upperBound(0, 1);
	double upperBernstein = b.upperBound();
	REQUIRE(upperBernstein <= upperCoarse);
	double lowerBernstein = b.lowerBound();

	for (int i = 0; i <= 100; ++i)
	{
		double x = i / 100.0;
		INFO("i=" << i << ", x=" << x);
		REQUIRE(f(x) >= lowerBernstein);
		REQUIRE(f(x) <= upperBernstein);
	}
}

TEST_CASE("Bernstein-Random", "[Math]")
{
	std::default_random_engine rnd(42);
	kernel::Polynomial<5, double> f;

	for (int i=0; i<50; ++i)
	{
		for (int j = 0; j <= 5; ++j)
			f[j] = std::uniform_real_distribution<double>(-2, +2)(rnd);
		INFO("i=" << i << ", f(x)=" << f);

		for (int i2=0; i2<20; ++i2)
		{
			double a = i2 == 0 ? 0.0 : std::uniform_real_distribution<double>(1e-4, 2)(rnd);
			double b = i2 == 0 ? 1.0 : (a+std::uniform_real_distribution<double>(1e-4, 2-a)(rnd));
			INFO("i2=" << i2 << ", interval=[" << a << ", " << b << "]");

			const auto f2 = f.transform(a, b - a);
			const auto bernstein = kernel::Bernstein<5, double>::fromPolynomial(f2);

			//check for equality
			for (int j = 0; j <= 10; ++j)
			{
				double x = j / 10.0;
				double x2 = a + x * (b - a);
				INFO("j=" << j << ", x=" << x);
				REQUIRE(f(x2) == Approx(f2(x)));
				REQUIRE(f(x2) == Approx(bernstein(x)).epsilon(1e-4));
			}

			//check bounds
			double maxValue = -FLT_MAX;
			for (int j = 0; j <= 100; ++j)
			{
				double x = j / 100.0;
				maxValue = fmax(maxValue, bernstein(x));
			}
			double maxBernstein = bernstein.upperBound();
			REQUIRE(maxValue <= maxBernstein);
		}
	}
}

template<int OrderQ, int OrderP>
void testBernsteinPolyExpPoly(std::default_random_engine& rnd)
{
	kernel::PolyExpPoly<OrderQ, double, OrderP, double> pq;
	std::normal_distribution<double> distr1;
	INFO("OrderQ=" << OrderQ << ", OrderP=" << OrderP);
	for (int j = 0; j < 20; ++j) {
		for (int i = 0; i <= OrderQ; ++i) pq.q[i] = distr1(rnd);
		for (int i = 0; i <= OrderP; ++i) pq.p[i] = distr1(rnd);
		INFO("j=" << j << ", pq = " << pq.q << " exp " << pq.p);

		for (int k=0; k<20; ++k)
		{
			double a1 = distr1(rnd);
			double b1 = distr1(rnd);
			double a2 = k == 0 ? 0 : std::min(a1, b1);
			double b2 = k == 0 ? 1 : std::max(a1, b1);
			INFO("k=" << k << ", a=" << a2 << ", b=" << b2);

			double boundBernstein = pq.absBoundBernstein(a2, b2);
			double boundNum = -FLT_MAX;
			static const int NUM_SAMPLES = 100;
			for (int i=0; i<=NUM_SAMPLES; ++i)
			{
				double x = a2 + (k / double(NUM_SAMPLES)) * (b2 - a2);
				boundNum = std::max(boundNum, std::abs(pq(x)));
			}
			REQUIRE(boundNum <= boundBernstein + 1e-6);
		}
	}
}

TEST_CASE("Bernstein-PolyExpPoly", "[Math]")
{
	std::default_random_engine rnd(42);
	testBernsteinPolyExpPoly<3, 3>(rnd);
	testBernsteinPolyExpPoly<5, 3>(rnd);
	testBernsteinPolyExpPoly<7, 3>(rnd);
	testBernsteinPolyExpPoly<12, 3>(rnd);
	testBernsteinPolyExpPoly<3, 4>(rnd);
	testBernsteinPolyExpPoly<5, 4>(rnd);
	testBernsteinPolyExpPoly<7, 4>(rnd);
	testBernsteinPolyExpPoly<12, 4>(rnd);
}
