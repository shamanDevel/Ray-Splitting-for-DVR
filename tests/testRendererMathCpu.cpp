#include <catch.hpp>
#include <random>

#include <renderer_math_cpu.h>

template<int N>
void testPolynomialAllRoots(std::default_random_engine& rnd)
{
	std::normal_distribution<double> distr;
	for (int i = 0; i < 20; ++i)
	{
		kernel::Polynomial<N, double> poly;
		kernel::Polynomial<N, std::complex<double>> cpoly;
		for (int j = 0; j < N + 1; ++j) {
			poly.coeff[j] = distr(rnd);
			cpoly.coeff[j] = poly.coeff[j];
		}
		INFO("i=" << i << ", poly=" << poly);

		const auto complexRoots = RendererMathCpu::roots(poly);
		INFO("roots: " << complexRoots);
		REQUIRE(complexRoots.size() == N);
		for (int j=0; j<N; ++j)
		{
			std::complex<double> root = complexRoots[j];
			std::complex<double> value = cpoly(root);
			INFO("x=" << root << ", value=" << value);
			CHECK(std::abs(value) < 1e-3);
		}

		const auto realRoots = RendererMathCpu::extractRealRoots(complexRoots, 1e-7f);
		for (size_t j=0; j<realRoots.size(); ++j)
		{
			double root = realRoots[j];
			double value = poly(root);
			INFO("x=" << root << ", value=" << value);
			CHECK(std::abs(value) < 1e-3);
		}
	}
}
TEST_CASE("Polynomial-AllRoots", "[MathCpu]")
{
	std::default_random_engine rnd(43);
	SECTION("N=3") { testPolynomialAllRoots<3>(rnd); }
	SECTION("N=4") { testPolynomialAllRoots<4>(rnd); }
	SECTION("N=5") { testPolynomialAllRoots<5>(rnd); }
	SECTION("N=8") { testPolynomialAllRoots<8>(rnd); }
}
