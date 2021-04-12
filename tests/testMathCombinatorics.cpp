#include <catch.hpp>
#include <array>
#include <iostream>
#include <type_traits>

#include <renderer_math.cuh>


TEST_CASE("Combinatorics - factorial",  "[Math]")
{
	static unsigned long long FACTORIALS[] = {
		1,1,2,6,24,120,720,5040,40320,362880,3628800,39916800 };

	for (int i=0; i<sizeof(FACTORIALS)/sizeof(unsigned long long); ++i)
	{
		INFO("i=" << i);
		unsigned long long expected = FACTORIALS[i];
		unsigned long long actual = kernel::Combinatorics::factorial(i);
		REQUIRE(expected == actual);
	}
}

TEST_CASE("Combinatorics - binomial", "[Math]")
{
	static int BINOMIALS[11][11]
	{
		{ 1 },
		{ 1,   1 },
		{ 1,   2,   1 },
		{ 1,   3,   3,   1 },
		{ 1,   4,   6,   4,   1 },
		{ 1,   5,  10,  10,   5,   1 },
		{ 1,   6,  15,  20,  15,   6,   1 },
		{ 1,   7,  21,  35,  35,  21,   7,   1 },
		{ 1,   8,  28,  56,  70,  56,  28,   8,   1 },
		{ 1,   9,  36,  84, 126, 126,  84,  36,   9,   1 },
		{ 1,  10,  45, 120, 210, 252, 210, 120,  45,  10,   1 },
	};
	for (int n=0; n<=10; ++n) for (int k=0; k<=n; ++k)
	{
		INFO("n=" << n << ", k=" << k);
		REQUIRE(BINOMIALS[n][k] == kernel::Combinatorics::binomial(n, k));
	}
}

TEST_CASE("Combinatorics - constexpr", "[Math]")
{
	//test constexpr for some examples
	static_assert(39916800 == kernel::Combinatorics::factorial(11));
	static_assert(120 == kernel::Combinatorics::binomial(10, 7));
}
