#include <catch.hpp>
#include <iostream>
#include <sstream>

#include <renderer_math.cuh>

TEST_CASE("Marmitt-SingleMulti", "[Math]")
{
	//compares single vs. multiple-root Marmitt

	std::default_random_engine rnd(42);
	std::uniform_real_distribution<float> distr(0, 1);

	const int NUM_TESTS = 1000;
	for (int test = 0; test < NUM_TESTS; ++test)
	{
		INFO("run " << test)
		{

			float vals[8];
			std::stringstream info;
			info << "Vertex values:";
			for (float& val : vals)
			{
				val = distr(rnd) * 2 - 1;
				info << " " << val;
			}
			info << std::endl;

			float3 entry = make_float3(distr(rnd), distr(rnd), distr(rnd));
			float3 exit = make_float3(distr(rnd), distr(rnd), distr(rnd));
			float3 dir = exit - entry;
			info << "Entry: " << entry.x << " " << entry.y << " " << entry.z << std::endl;
			info << "Exit: " << exit.x << " " << exit.y << " " << exit.z << std::endl;
			info << "Direction: " << dir.x << " " << dir.y << " " << dir.z << std::endl;
			INFO(info.str());

			float tExit = distr(rnd) + 0.0001f;
			INFO("tExit: " << tExit);
			dir = dir / tExit;

			float4 factors = kernel::CubicPolynomial<float>::getFactors(
				vals, entry, dir);

			float firstHitTime;
			bool hasHit = kernel::Marmitt<3, true>::eval(
				vals, factors, 0, entry, dir, 0, tExit, firstHitTime);
			float allHitTimes[3];
			int numHits = kernel::Marmitt<3, true>::evalAll(
				vals, factors, 0, entry, dir, 0, tExit, allHitTimes);
			INFO("num hits: " << numHits);
			
			REQUIRE(hasHit == (numHits>0));
			if (hasHit)
				REQUIRE(firstHitTime == Approx(allHitTimes[0]).epsilon(1e-8));
		}
	}
}

TEST_CASE("Marmitt-Iso", "[Math]")
{
	//Tests if setting the isovalue works

	std::default_random_engine rnd(42);
	std::uniform_real_distribution<float> distr(0, 1);

	const int NUM_TESTS = 1000;
	for (int test = 0; test < NUM_TESTS; ++test)
	{
		INFO("run " << test)
		{

			float vals[8];
			std::stringstream info;
			info << "Vertex values:";
			for (float& val : vals)
			{
				val = distr(rnd) * 2 - 1;
				info << " " << val;
			}
			info << std::endl;

			float3 entry = make_float3(distr(rnd), distr(rnd), distr(rnd));
			float3 exit = make_float3(distr(rnd), distr(rnd), distr(rnd));
			float3 dir = exit - entry;
			info << "Entry: " << entry.x << " " << entry.y << " " << entry.z << std::endl;
			info << "Exit: " << exit.x << " " << exit.y << " " << exit.z << std::endl;
			info << "Direction: " << dir.x << " " << dir.y << " " << dir.z << std::endl;
			INFO(info.str());

			float tExit = distr(rnd) + 0.0001f;
			INFO("tExit: " << tExit);
			dir = dir / tExit;

			float isovalue = distr(rnd) * 2 - 1;
			INFO("Isovalue: " << isovalue);

			float4 factors = kernel::CubicPolynomial<float>::getFactors(
				vals, entry, dir);

			float firstHitTime;
			bool hasHit = kernel::Marmitt<15, true>::eval(
				vals, factors, isovalue, entry, dir, 0, tExit, firstHitTime);
			float allHitTimes[3];
			int numHits = kernel::Marmitt<15, true>::evalAll(
				vals, factors, isovalue, entry, dir, 0, tExit, allHitTimes);
			INFO("num hits: " << numHits);

			REQUIRE(hasHit == (numHits > 0));
			if (hasHit)
				REQUIRE(firstHitTime == Approx(allHitTimes[0]).epsilon(1e-8));
			for (int i=0; i<numHits; ++i)
			{
				float val = kernel::lerp3D(vals, entry + dir * allHitTimes[i]);
				REQUIRE(val == Approx(isovalue).epsilon(1e-3));
			}
		}
	}
}