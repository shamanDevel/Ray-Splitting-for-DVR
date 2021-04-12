#include <catch.hpp>
#include <array>
#include <iostream>
#include <type_traits>

#include <renderer_math.cuh>

template <class _Ty>
struct is_aggregate : std::integral_constant<bool, __is_aggregate(_Ty)> {};

TEST_CASE("Polynomial-Eval", "[Math]")
{
	STATIC_REQUIRE(is_aggregate<kernel::Polynomial<2, float> >::value);
	SECTION("f(x)=5")
	{
		kernel::Polynomial<0, float> p{ 5 };
		REQUIRE(p.eval(0.5) == 5);
		REQUIRE(p(0.5f) == 5);
		REQUIRE(p.evalAtOne() == 5);
		REQUIRE(p.evalAtZero() == 5);
		REQUIRE(p.integrate()(0.5) == 2.5);
	}

	SECTION("f(x)=5+3x-8x^2")
	{
		kernel::Polynomial<2, float> p{ 5, 3, -8 };
		REQUIRE(p.eval(0.5) == Approx(4.5));
		REQUIRE(p(0.5f) == Approx(4.5));
		REQUIRE(p.evalAtOne() == Approx(0));
		REQUIRE(p.evalAtZero() == Approx(5));
		REQUIRE(p.derivative().degree() == 1);
		REQUIRE(p.derivative()(0) == Approx(3));
		REQUIRE(p.derivative()(3/16.f) == Approx(0));
		REQUIRE(p.integrate().degree() == 3);
		REQUIRE(p.integrate()(6) - p.integrate()(-2) == Approx(-1528/3.f));
	}
}

TEST_CASE("Polynomial-Lerp", "[Math]")
{
	kernel::Polynomial<3, float> p{ 2, 0.5f, -0.4f, 1 };
	float xA = 0.5f;
	float2 dA{ 2, 3 };
	float xB = 4.0f;
	float2 dB{ 7, 9 };

	kernel::Polynomial<3, float2> pHat = p.lerp(p(xA), dA, p(xB), dB);

	REQUIRE(pHat(make_float2(xA)).x == Approx(dA.x));
	REQUIRE(pHat(make_float2(xA)).y == Approx(dA.y));
	REQUIRE(pHat(make_float2(xB)).x == Approx(dB.x));
	REQUIRE(pHat(make_float2(xB)).y == Approx(dB.y));
	
	int N = 20;
	for (int i=0; i<=N; ++i)
	{
		INFO("i=" << i);
		float x = xA + (xB - xA) * i / static_cast<float>(N);
		float t = p(x);
		
		float2 dOrig = dA + (dB - dA) * (t - p(xA)) / (p(xB) - p(xA));
		
		float2 d = pHat(make_float2(x));
		REQUIRE(dOrig.x == Approx(d.x));
		REQUIRE(dOrig.y == Approx(d.y));
	}
}

TEST_CASE("Polynomial-Multiplication", "[Math]")
{
	kernel::Polynomial<3, double> a{ 0.5, -2.5, 1.3, -0.2 };
	kernel::Polynomial<4, double> b{ 1.5, 0.3, -0.5, 0.8, 1.2 };
	kernel::Polynomial<7, double> ab = a * b;
	for (double x = -5; x <= +5; x += 0.125)  // NOLINT(cert-flp30-c)
	{
		REQUIRE(ab(x) == Approx(a(x) * b(x)));
	}
}

TEST_CASE("Polynomial-Addition", "[Math]")
{
	kernel::Polynomial<3, double> a{ 0.5, -2.5, 1.3, -0.2 };
	kernel::Polynomial<4, double> b{ 1.5, 0.3, -0.5, 0.8, 1.2 };
	kernel::Polynomial<4, double> ab = a + b;
	for (double x = -5; x <= +5; x += 0.125)  // NOLINT(cert-flp30-c)
	{
		REQUIRE(ab(x) == Approx(a(x) + b(x)));
	}
}

template<int N>
void testPolynomialUpperBound(std::default_random_engine& rnd)
{
	std::normal_distribution<double> distr;
	static const int Samples = 50;
	for (int i = 0; i<20; ++i)
	{
		kernel::Polynomial<N, double> poly;
		for (int j = 0; j < N + 1; ++j) poly.coeff[j] = distr(rnd);
		INFO("i=" << i << ", poly=" << poly);
		for (int j=0; j<20; ++j)
		{
			double a1 = distr(rnd), b1 = distr(rnd);
			double a2 = std::min(a1, b1), b2 = std::max(a1, b1);
			double bound = poly.upperBound(a2, b2);
			INFO("j=" << j << ", a=" << a2 << ", b=" << b2 << " -> bound=" << bound);
			double numBound = 0;
			for (int k=0; k<=Samples; ++k)
			{
				double x = a2 + (k / double(Samples)) * (b2 - a2);
				numBound = std::max(numBound, std::abs(poly.eval(x)));
			}
			REQUIRE(numBound - 1e-5 <= bound);
		}
	}
}
TEST_CASE("Polynomial-UpperBound", "[Math]")
{
	std::default_random_engine rnd(42);
	SECTION("N=3") { testPolynomialUpperBound<3>(rnd); }
	SECTION("N=4") { testPolynomialUpperBound<4>(rnd); }
	SECTION("N=5") { testPolynomialUpperBound<5>(rnd); }
	SECTION("N=8") { testPolynomialUpperBound<8>(rnd); }
}

TEST_CASE("Polynomial-transform", "[Math]")
{
	kernel::Polynomial<3, double> f{ 0.5, -2.5, 1.3, -0.2 };;
	double a = 0.2, b = 0.8;
	auto g = f.transform(a, b - a);

	REQUIRE(g.coeff[3] == Approx(-27.0 / 625.0));   //*x^3
	REQUIRE(g.coeff[2] == Approx(531.0 / 1250.0));  //*x^2
	REQUIRE(g.coeff[1] == Approx(-1503.0 / 1250.0));//*x^1
	REQUIRE(g.coeff[0] == Approx(63.0 / 1250.0));   //*x^0

	REQUIRE(g(0) == Approx(f(a)));
	REQUIRE(g(1) == Approx(f(b)));
	for (int i = 0; i <= 10; ++i)
	{
		double x = i / 10.0;
		REQUIRE(g(x) == Approx(f(a + x * (b - a))));
	}
}

TEST_CASE("Polynomial-transform-random", "[Math]")
{
	kernel::Polynomial<3, double> f{ 0.5, -2.5, 1.3, -0.2 };
	std::default_random_engine rnd(42);
	for (int j=0; j<20; ++j)
	{
		double a1 = std::uniform_real_distribution<double>(0,1)(rnd);
		double b1 = std::uniform_real_distribution<double>(0,1)(rnd);
		double a2 = std::min(a1, b1), b2 = std::max(a1, b1);
		if (j == 0) { a2 = 0; b2 = 1; }
		INFO("j=" << j << ", a=" << a2 << ", b=" << b2);

		auto g = f.transform(a2, b2 - a2);
		REQUIRE(g(0) == Approx(f(a2)));
		REQUIRE(g(1) == Approx(f(b2)));
		for (int i = 0; i <= 10; ++i)
		{
			double x = i / 10.0;
			REQUIRE(g(x) == Approx(f(a2 + x * (b2 - a2))));
		}
	}
}
