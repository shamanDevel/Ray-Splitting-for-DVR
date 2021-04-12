#include <catch.hpp>
#include <iostream>
#include <sstream>

#include <cuda_runtime.h>

template<typename T>
static T tex3D(cudaTextureObject_t tex, float x, float y, float z) { throw std::exception("not allowed in host code"); }

template<typename T>
static T tex1D(cudaTextureObject_t tex, float x) { throw std::exception("not allowed in host code"); }

#include <renderer_math.cuh>
#include <renderer_impl_dvr.cuh>

/**
 * Test implementation
 */
struct DvrVoxelEvalInterval_Test
{
	const float4 poly;
	struct ControlPoint
	{
		float isovalue;
		float data;
		int numIntersections;
		float intersections[3];
	};
	const std::vector<ControlPoint> controlPoints;
	std::vector<bool> intersectionsComputed;
	
	struct Interval
	{
		float tEntry, tExit;
		float dataEntry, dataExit;
		kernel::Polynomial<3, float> poly;
	};

	DvrVoxelEvalInterval_Test(float4 poly, const std::vector<ControlPoint> controlPoints)
		: poly(poly), controlPoints(controlPoints)
		, intersectionsComputed(controlPoints.size(), false)
	{}
	
private:

	__host__ __device__ int searchInterval(float density) const
	{
		int i;
		//for now, a simple linear search is used
		for (i = 0; i < controlPoints.size() - 1; ++i)
			if (controlPoints[i + 1].isovalue > density) break;
		return i;
	}

	__host__ __device__ float query(float density, int lowerInterval) const
	{
		const float pLow = controlPoints[lowerInterval].isovalue;
		const float pHigh = controlPoints[lowerInterval + 1].isovalue;
		REQUIRE(pLow <= density);
		REQUIRE(pHigh >= density);
		const float frac = (density - pLow) / (pHigh - pLow);
		return (1 - frac) * controlPoints[lowerInterval].data + frac * controlPoints[lowerInterval + 1].data;
	}
	
	/**
	 * \brief Computes the interval
	 *  \f$ \int_{tEntry}^{tExit} L(x) exp(O(x)) dx $\f
	 * where L is the emission and O the absorption.
	 * The mapping from x to density is given by the cubic polynomial 'poly'.
	 * The transfer function is linear in the density, given by
	 *    dataEntry = (L(tf(tEntry)), O(tf(tEntry)))
	 * and
	 *    dataExit = (L(tf(tExit)), O(tf(tExit))) .
	 *
	 * The results are front-to-back blended with rgbBufferOut and oBufferOut.
	 * \param poly
	 * \param tEntry
	 * \param tExit
	 * \param dataEntry
	 * \param dataExit
	 */
	[[nodiscard]] static Interval emitInterval(
		const float4& poly, float tEntry, float tExit,
		float densityEntry, float densityExit, float dataEntry, float dataExit)
	{
		auto density = kernel::float4ToPoly<float>(poly);
		auto dataPoly = density.lerp(
			densityEntry, dataEntry,
			densityExit, dataExit);
		return { tEntry, tExit, dataEntry, dataExit, dataPoly };
	}

public:
	[[nodiscard]] int evalAll(int index, float tEntry, float tExit, float* intersections)
	{
		REQUIRE(index >= 0);
		REQUIRE(index < int(controlPoints.size()));
		REQUIRE_FALSE(intersectionsComputed[index]);
		intersectionsComputed[index] = true;
		if (controlPoints[index].numIntersections>0)
		{
			REQUIRE(tEntry < controlPoints[index].intersections[0]);
			for (int j = 0; j < controlPoints[index].numIntersections; ++j)
				intersections[j] = controlPoints[index].intersections[j];
		}
		return controlPoints[index].numIntersections;
	}
	
	std::vector<Interval> call(float tExit)
	{
		std::vector<Interval> intervals;

		//find initial control point interval
		float timeEntry = 0;
		float densityEntry = kernel::CubicPolynomial<float>::evalCubic(poly, timeEntry);
		int currentIndex = searchInterval(densityEntry);
		float dataEntry = query(densityEntry, currentIndex);

		//TODO: early out if the exit point is in the same interval
		//and the opacity in that interval is always zero

		//store intersections and reuse them
		float intersections[TF_MAX_CONTROL_POINTS][3];
		int numIntersections[TF_MAX_CONTROL_POINTS];
#pragma unroll
		for (int i = 0; i < TF_MAX_CONTROL_POINTS; ++i) numIntersections[i] = -1;
		int currentIntersection[TF_MAX_CONTROL_POINTS];

		//compute first intersections
		numIntersections[currentIndex] = evalAll(currentIndex, timeEntry, tExit,
			intersections[currentIndex]);
		currentIntersection[currentIndex] = 0;
		numIntersections[currentIndex+1] = evalAll(currentIndex+1, timeEntry, tExit,
			intersections[currentIndex+1]);
		currentIntersection[currentIndex+1] = 0;

		int indexPrevious = -1;
		while (true) // breaks if no more intersections are found
		{
			//debug - invariants
			REQUIRE(currentIndex >= 0);
			REQUIRE(currentIndex+1 < int(controlPoints.size()));

			bool hasLower = numIntersections[currentIndex] > currentIntersection[currentIndex];
			float timeLower = hasLower
				? intersections[currentIndex][currentIntersection[currentIndex]] : tExit + 1;
			bool hasUpper = numIntersections[currentIndex+1] > currentIntersection[currentIndex+1];
			float timeUpper = hasUpper
				? intersections[currentIndex+1][currentIntersection[currentIndex+1]] : tExit + 1;

			if (tExit < timeLower && tExit < timeUpper)
			{
				//exit the voxel first
				const float densityExit = kernel::CubicPolynomial<float>::evalCubic(poly, tExit);
				const int controlPoint = searchInterval(densityExit);
				const float dataExit = query(densityExit, controlPoint);
				intervals.push_back(emitInterval(poly, timeEntry, tExit,
					densityEntry, densityExit,
					dataEntry, dataExit));
				break;
			}
			else if (timeLower < timeUpper)
			{
				REQUIRE(hasLower);
				float dataExit;
				float densityExit;
				if (currentIndex == indexPrevious)
				{
					densityExit = controlPoints[currentIndex+1].isovalue;
					dataExit = controlPoints[currentIndex+1].data;
				}
				else
				{
					densityExit = controlPoints[currentIndex].isovalue;
					dataExit = controlPoints[currentIndex].data;
				}
				intervals.push_back(emitInterval(poly, timeEntry, timeLower,
					densityEntry, densityExit,
					dataEntry, dataExit));
				densityEntry = densityExit;
				dataEntry = controlPoints[currentIndex].data;
				indexPrevious = currentIndex;
				timeEntry = timeLower;
				currentIntersection[currentIndex]++; //pop the intersection
				currentIndex--;
				if (numIntersections[currentIndex]==-1)
				{
					numIntersections[currentIndex] = evalAll(
						currentIndex, timeEntry, tExit,
						intersections[currentIndex]);
					currentIntersection[currentIndex] = 0;
				}
			}
			else
			{
				//upper poly
				REQUIRE(hasUpper);
				float dataExit;
				float densityExit;
				if (currentIndex+1 == indexPrevious)
				{
					densityExit = controlPoints[currentIndex].isovalue;
					dataExit = controlPoints[currentIndex].data;
				}
				else
				{
					densityExit = controlPoints[currentIndex+1].isovalue;
					dataExit = controlPoints[currentIndex+1].data;
				}
				intervals.push_back(emitInterval(poly, timeEntry, timeUpper,
					densityEntry, densityExit,
					dataEntry, dataExit));
				densityEntry = densityExit;
				dataEntry = controlPoints[currentIndex + 1].data;
				indexPrevious = currentIndex+1;
				timeEntry = timeUpper;
				currentIntersection[currentIndex+1]++; //pop the intersection
				currentIndex++;
				if (numIntersections[currentIndex+1] == -1)
				{
					numIntersections[currentIndex+1] = evalAll(
						currentIndex+1, timeEntry, tExit,
						intersections[currentIndex+1]);
					currentIntersection[currentIndex+1] = 0;
				}
			}
		}

		return intervals;
	}
};


TEST_CASE("DvrVoxelEval-FixedExample", "[Dvr]")
{
	float4 poly{ 0.1, -0.52, 0.7, 0.08 };
	std::vector<DvrVoxelEvalInterval_Test::ControlPoint> points{
		{0, 0, 0, {0.0f,0.0f,0.0f}},
		{0.25, 0, 3, {0.31f, 1.7422f, 3.1478f}},
		{0.35, 0.7, 3, {0.70425, 1.14376, 3.3520}},
		{0.45, 0.2, 1, {3.5045, 0, 0}},
		{0.675, 0.3, 1, {3.7588, 0, 0}},
		{0.725, 0, 1, {3.8061, 0, 0}},
		{1,0,0,{0,0,0}}
	};
	auto evaluator = DvrVoxelEvalInterval_Test(poly, points);
	auto intervals = evaluator.call(4.0f);

	//printing
	std::cout << "Num intervals: " << intervals.size() << "\n";
	for (size_t i=0; i<intervals.size(); ++i)
	{
		std::cout << "[" << i << "]: t=(" << intervals[i].tEntry << ", " << intervals[i].tExit
			<< "), d=(" << intervals[i].dataEntry << ", " << intervals[i].dataExit << ")"
			<< ": f(x)=" << intervals[i].poly << "\n";
	}

	
}

TEST_CASE("DvrVoxelEval-Random", "[Dvr]")
{
	std::default_random_engine rnd(42);
	std::uniform_real_distribution<float> distr(0, 1);

	kernel::RendererDeviceSettings settings;
	settings.opacityScaling = 1.0f;
	settings.realMinDensity = 0.0f;
	settings.voxelSize = { 1,1,1 };
	std::vector<float> tfPositions = { 0.0f, 0.45f, 0.5f, 0.55f, 1.0f};
	std::vector<float4> tfData = { {0,0,0,0}, {1,0,0,0,}, {0,1,0,1}, {0,0,1,0}, {0,0,0,0} };
	for (size_t i=0; i<tfPositions.size(); ++i)
	{
		settings.tfPoints.positions[i] = tfPositions[i];
		settings.tfPoints.values[i] = tfData[i];
	}
	settings.tfPoints.numPoints = tfPositions.size();
	
	
	const int NUM_TESTS = 1;// 10000;
	for (int test = 0; test < NUM_TESTS; ++test)
	{
		INFO("run " << test)
		{

			float vals[8];
			std::stringstream info;
			info << "Vertex values:";
			for (float& val : vals)
			{
				val = distr(rnd);
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

			float4 rgbDensity = { 0,0,0,0 };
			float3 normalOut;
			float depthOut;

			bool hasData = kernel::DvrVoxelEvalInterval<kernel::DvrIntervalEvaluatorSimple, 15>
			::call(vals, entry, dir, tExit, settings, 0, rgbDensity, normalOut, depthOut, false);
		}
	}
}