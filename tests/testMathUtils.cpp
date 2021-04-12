#include <catch.hpp>
#include <array>
#include <iostream>
#include <type_traits>

#include <renderer_math.cuh>


TEST_CASE("Lerp3D-DErivatives", "[Math]")
{
	std::default_random_engine rnd(42);
	std::normal_distribution<float> distr1;
	std::uniform_real_distribution<float> distr2(0, 1);

	static const float DELTA = 1e-3;
	static const float EPSILON = 1e-1;
	using namespace kernel;
	
	for (int i1=0; i1<50; ++i1)
	{
		INFO("i1=" << i1);
		float vals[8];
		for (int i = 0; i < 8; ++i) vals[i] = distr1(rnd);

		for (int i2=0; i2<50; ++i2)
		{
			float3 p = make_float3(
				distr2(rnd), distr2(rnd), distr2(rnd));
			INFO("i2=" << i2 << ", p=" << p.x << ", " << p.y << ", " << p.z);

			INFO("val=" << lerp3D(vals, p));
			float3 gradNumeric;
			gradNumeric.x = 
				(lerp3D(vals, p + make_float3(DELTA, 0, 0)) - lerp3D(vals, p - make_float3(DELTA, 0, 0))) /
				(2 * DELTA);
			gradNumeric.y =
				(lerp3D(vals, p + make_float3(0, DELTA, 0)) - lerp3D(vals, p - make_float3(0, DELTA, 0))) /
				(2 * DELTA);
			gradNumeric.z =
				(lerp3D(vals, p + make_float3(0, 0, DELTA)) - lerp3D(vals, p - make_float3(0, 0, DELTA))) /
				(2 * DELTA);

			float3 gradAnalytic = lerp3DDerivatives(vals, p);

			REQUIRE(gradNumeric.x == Approx(gradAnalytic.x).epsilon(EPSILON));
			REQUIRE(gradNumeric.y == Approx(gradAnalytic.y).epsilon(EPSILON));
			REQUIRE(gradNumeric.z == Approx(gradAnalytic.z).epsilon(EPSILON));
		}
	}
}
