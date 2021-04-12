#pragma once

#include "renderer_math.cuh"

/// <summary>
/// Additional math helpers for isosurface rendering.
///
/// Especially, this file contains higher-order interpolation and solvers.
/// 
/// </summary>

namespace kernel
{
	enum class TricubicFactorAlgorithm
	{
		LOOP,
		EXPLICIT
	};
	/**
	 * \brief Utilities for tricubic interpolation.
	 * Especially, this method contains tools to compute the nonic
	 * polynomial for the tricubic interpolation.
	 * 
	 * \tparam Scalar_t the scalar type (float or double)
	 */
	template<typename Scalar_t>
	struct TricubicInterpolation
	{
		typedef Polynomial<9, Scalar_t> poly_t;

	private:
		typedef Polynomial<3, Scalar_t> basis_t;
		
		// for 0<=x<1:
		//N0(x) = 2/3 - 1/2*(x^2) * (2-x);

		// for 1<=x<2:
		//N1(x) = 1/6*(2-x)^3;

		// ray:
		//ray(t, e, d) = e + t*d

		// N0(ray(t,e,d))
		static __host__ __device__ __forceinline__ basis_t N0M(
			const Scalar_t e, const Scalar_t d,
			const Scalar_t e2, const Scalar_t d2, 
			const Scalar_t e3, const Scalar_t d3)
		{
			return basis_t{
				e3 * 0.5f - e2 + (2 / 3.0f),
				1.5f * d * e2 - 2 * d * e,
				(1.5f * d2 * e) - d2,
				d3 * 0.5f
			};
		}

		// N0(1-ray(t,e,d))
		static __host__ __device__ __forceinline__ basis_t N0P(
			const Scalar_t e, const Scalar_t d,
			const Scalar_t e2, const Scalar_t d2,
			const Scalar_t e3, const Scalar_t d3)
		{
			return basis_t{
				0.5f * (e2 - e3 + e) + 1/6.0f,
				0.5f * (d -  (3 * d * e2)) + d * e,
				0.5f * (d2 - (3 * d2 * e)),
				-d3 * 0.5f
			};
		}

		// N1(1+ray(t,e,d))
		static __host__ __device__ __forceinline__ basis_t N1M(
			const Scalar_t e, const Scalar_t d,
			const Scalar_t e2, const Scalar_t d2,
			const Scalar_t e3, const Scalar_t d3)
		{
			return basis_t{
				(1/6.0f) * (1 - e3) + 0.5f * (e2 - e),
				d * (e - 0.5f - 0.5f * e2),
				0.5f * d2 * (1 - e),
				-d3 / 6
			};
		}

		// N1(2-ray(t,e,d))
		static __host__ __device__ __forceinline__ basis_t N1P(
			const Scalar_t e, const Scalar_t d,
			const Scalar_t e2, const Scalar_t d2,
			const Scalar_t e3, const Scalar_t d3)
		{
			return basis_t{
				e3 / 6,
				(d * e2) / 2,
				(d2 * e) / 2,
				d3 / 6
			};
		}

	public:
		
		/**
		 * \brief Returns the nonic polynomial for the ray r(t)=entry+t*dir,
		 * traversing the voxel with corner values given by 'vals' and accessing the
		 * tri-cubic interpolated values.
		 *
		 * This variation uses an algorithm using loops and
		 * polygonial multiplication.
		 *
		 * The ray positions are in [0,1]^3 for the course of the whole traversal
		 * of the voxel.
		 * The voxel values are given with x fastest, z slowest:
		 * vals[x + 4*(y + 4*z)], 0<=x,y,z<4
		 * 
		 * \param vals the 64-neighborhood around the ray segment
		 * \param entry the ray entry
		 * \param dir the ray direction
		 * \return the nonic polynomial
		 */
		static __host__ __device__ poly_t getFactors(
			const float vals[64], const float3& entry, const float3& dir,
			integral_constant<TricubicFactorAlgorithm, TricubicFactorAlgorithm::LOOP>)
		{
			const Scalar_t ex = entry.x; Scalar_t ex2 = ex * ex; Scalar_t ex3 = ex * ex2;
			const Scalar_t ey = entry.y; Scalar_t ey2 = ey * ey; Scalar_t ey3 = ey * ey2;
			const Scalar_t ez = entry.z; Scalar_t ez2 = ez * ez; Scalar_t ez3 = ez * ez2;

			const Scalar_t dx = dir.x; Scalar_t dx2 = dx * dx; Scalar_t dx3 = dx * dx2;
			const Scalar_t dy = dir.y; Scalar_t dy2 = dy * dy; Scalar_t dy3 = dy * dy2;
			const Scalar_t dz = dir.z; Scalar_t dz2 = dz * dz; Scalar_t dz3 = dz * dz2;

			const basis_t Nx[4] = {
				N1M(ex, dx, ex2, dx2, ex3, dx3),
				N0M(ex, dx, ex2, dx2, ex3, dx3),
				N0P(ex, dx, ex2, dx2, ex3, dx3),
				N1P(ex, dx, ex2, dx2, ex3, dx3)
			};
			const basis_t Ny[4] = {
				N1M(ey, dy, ey2, dy2, ey3, dy3),
				N0M(ey, dy, ey2, dy2, ey3, dy3),
				N0P(ey, dy, ey2, dy2, ey3, dy3),
				N1P(ey, dy, ey2, dy2, ey3, dy3)
			};
			const basis_t Nz[4] = {
				N1M(ez, dz, ez2, dz2, ez3, dz3),
				N0M(ez, dz, ez2, dz2, ez3, dz3),
				N0P(ez, dz, ez2, dz2, ez3, dz3),
				N1P(ez, dz, ez2, dz2, ez3, dz3)
			};

			poly_t out = { 0 };
			for (int z=0; z<4; ++z)
			{
				const auto& currentNz = Nz[z];
				for (int y=0; y<4; ++y)
				{
					const auto currentNyz = currentNz * Ny[y];
					for (int x=0; x<4; ++x)
					{
						const auto currentNxyz = currentNyz * Nx[x];
						out += currentNxyz * vals[x + 4 * (y + 4 * z)];
					}
				}
			}

			return out;
		}

		/**
		 * \brief Returns the nonic polynomial for the ray r(t)=entry+t*dir,
		 * traversing the voxel with corner values given by 'vals' and accessing the
		 * tri-cubic interpolated values.
		 *
		 * This variation uses an algorithm where the factors
		 * are analytically solved in Matlab.
		 * (Huge source code)
		 *
		 * The ray positions are in [0,1]^3 for the course of the whole traversal
		 * of the voxel.
		 * The voxel values are given with x fastest, z slowest:
		 * vals[x + 4*(y + 4*z)], 0<=x,y,z<4
		 *
		 * \param vals the 64-neighborhood around the ray segment
		 * \param entry the ray entry
		 * \param dir the ray direction
		 * \return the nonic polynomial
		 */
		static __host__ __device__ poly_t getFactors(
			const float vals[64], const float3& entry, const float3& dir,
			integral_constant<TricubicFactorAlgorithm, TricubicFactorAlgorithm::EXPLICIT>)
		{
#include "renderer_math_iso_TricubicFactors.inl"
		}

		// instance methods
	private:
		basis_t Nx[4];
		basis_t Ny[4];
		basis_t Nz[4];

	public:
		__host__ __device__ __inline__ TricubicInterpolation(const float3& entry, const float3& dir)
		{
			const Scalar_t ex = entry.x; Scalar_t ex2 = ex * ex; Scalar_t ex3 = ex * ex2;
			const Scalar_t ey = entry.y; Scalar_t ey2 = ey * ey; Scalar_t ey3 = ey * ey2;
			const Scalar_t ez = entry.z; Scalar_t ez2 = ez * ez; Scalar_t ez3 = ez * ez2;

			const Scalar_t dx = dir.x; Scalar_t dx2 = dx * dx; Scalar_t dx3 = dx * dx2;
			const Scalar_t dy = dir.y; Scalar_t dy2 = dy * dy; Scalar_t dy3 = dy * dy2;
			const Scalar_t dz = dir.z; Scalar_t dz2 = dz * dz; Scalar_t dz3 = dz * dz2;

			Nx[0] = N1M(ex, dx, ex2, dx2, ex3, dx3);
			Nx[1] = N0M(ex, dx, ex2, dx2, ex3, dx3);
			Nx[2] = N0P(ex, dx, ex2, dx2, ex3, dx3);
			Nx[3] = N1P(ex, dx, ex2, dx2, ex3, dx3);
			
			Ny[0] = N1M(ey, dy, ey2, dy2, ey3, dy3);
			Ny[1] = N0M(ey, dy, ey2, dy2, ey3, dy3);
			Ny[2] = N0P(ey, dy, ey2, dy2, ey3, dy3);
			Ny[3] = N1P(ey, dy, ey2, dy2, ey3, dy3);
			
			Nz[0] = N1M(ez, dz, ez2, dz2, ez3, dz3);
			Nz[1] = N0M(ez, dz, ez2, dz2, ez3, dz3);
			Nz[2] = N0P(ez, dz, ez2, dz2, ez3, dz3);
			Nz[3] = N1P(ez, dz, ez2, dz2, ez3, dz3);
		}

		__host__ __device__ __inline__ Scalar_t call(const Scalar_t vals[64], Scalar_t t) const
		{
			Scalar_t out = 0;
			for (int z = 0; z < 4; ++z)
			{
				const Scalar_t coeff = Nz[z](t);
				for (int y = 0; y < 4; ++y)
				{
					const Scalar_t coeff2 = coeff * Ny[y](t);
					for (int x = 0; x < 4; ++x)
					{
						const Scalar_t coeff3 = coeff2 * Nx[x](t);
						out += coeff3 * vals[x + 4 * (y + 4 * z)];
					}
				}
			}
			return out;
		}
	};


	/**
	 * \brief General sphere tracing function.
	 * It solves for the first root in the interval [0, tMax]
	 * of the function "density".
	 * The bounds functor "bounds" provides a Lipschitz bound
	 * in [0, tMax].
	 *
	 * DensityFunctor_t must provide a method
	 * <code>Scalar_t operator()(const Scalar_t& x) const</code>.
	 * BoundsFunctor_t must provide a method
	 * <code>Scalar_t operator()(const DensityFunctor_t& f, Scalar_t tMax) const</code>.
	 * 
	 * \tparam DensityFunctor_t the function that queries the
	 *   density at time t of the ray. Usually, a \ref Polynomial
	 * \tparam BoundsFunctor_t the function to compute Lipschitz bounds
	 * \tparam Scalar_t the scalar type. If DensityFunctor_t is
	 *   an instance of \ref Polynomial, the scalar type can be automatically
	 *   deduced.
	 * \param density 
	 * \param tMax
	 * \param bounds 
	 * \param epsilon 
	 * \return 
	 */
	template<
		typename DensityFunctor_t,
		typename BoundsFunctor_t,
		typename Scalar_t = typename DensityFunctor_t::value_type>
	HD pair<Scalar_t, int> sphereTrace(
		const DensityFunctor_t& density,
		Scalar_t tMax, 
		const BoundsFunctor_t& bounds = BoundsFunctor_t(),
		Scalar_t epsilon = 1e-5,
		bool debug = false)
	{
		const Scalar_t L = bounds(density, tMax);
#ifndef KERNEL_NO_DEBUG
		if (debug) {
			printf("Upper bound with tMax=%.4f: L=%.4f\n", tMax, L);
		}
#endif
		Scalar_t t = 0;
		int numEval = 0;
		while (t < tMax)
		{
			Scalar_t fatT = density(t);
			numEval++;
			Scalar_t distance = fabsf(fatT / L);
#ifndef KERNEL_NO_DEBUG
			if (debug) {
				printf("  Sample t=%.6f -> d=%.6f, dist=%.6f\n", t, fatT, distance);
			}
#endif
			t += distance;
			if (distance <= epsilon)
				return { t, numEval };
		}
		return {-1, numEval};
	}

	/**
	 * \brief Simple polynomial bounds for \ref sphereTrace.
	 */
	struct SimpleBound
	{
		template<int Order, typename Scalar_t>
		HD Scalar_t operator()(
			const Polynomial<Order, Scalar_t>& f, const Scalar_t& tMax) const
		{
			return f.derivative().upperBound(Scalar_t(0), tMax);
		}
	};

	/**
	 * \brief More precise, but more expensive Bernstein bounds
	 * for \ref sphereTrace.
	 */
	struct BernsteinBound
	{
		template<int Order, typename Scalar_t>
		HD Scalar_t operator()(
			const Polynomial<Order, Scalar_t>& f, const Scalar_t& tMax) const
		{
			static_assert(Order == 9, "Bernstein Bounds are currently only implemented for Order=9");

			const auto fPrime01 = f.derivativeAndTransformTo01(tMax);

			//the Bernstein coefficients b_i from the
			//polynomial coefficients c_j are defined as
			//b_i = \sum_{j=0}^i (nPr(i,j)/nPr(n,j) * c_j)

			//I'm too lazy to find a nice way to compute the nPr-fractions
			//in compile time for any n,i,js. Just fix n to 8 (Order=9)

			const Scalar_t* gt = fPrime01.coeff;
			//BC0
			float maxCoeff = fabsf(gt[0]);
			//BCN
			float coeff = gt[0] + gt[1] + gt[2] + gt[3] + gt[4] + gt[5] + gt[6] + gt[7] + gt[8];
			coeff = fabsf(coeff);
			if (coeff > maxCoeff)
				maxCoeff = coeff;
			//BC1
			coeff = gt[0] + (1.0f / 8.0f) * gt[1];
			coeff = fabsf(coeff);
			if (coeff > maxCoeff)
				maxCoeff = coeff;
			//BC2
			coeff = gt[0] + 0.25f * gt[1] + (1.0f / 28.0f) * gt[2];
			coeff = fabsf(coeff);
			if (coeff > maxCoeff)
				maxCoeff = coeff;
			//BC3
			coeff = gt[0] + (3.0f / 8.0f) * gt[1] + (3.0f / 28.0f) * gt[2] + (1.0f / 56.0f) * gt[3];
			coeff = fabsf(coeff);
			if (coeff > maxCoeff)
				maxCoeff = coeff;
			//BC4
			coeff = gt[0] + 0.5f * gt[1] + (3.0f / 14.0f) * gt[2] + (1.0f / 14.0f) * gt[3] + (1.0f / 70.0f) * gt[4];
			coeff = fabsf(coeff);
			if (coeff > maxCoeff)
				maxCoeff = coeff;
			//BC5
			coeff = gt[0] + (5.0f / 8.0f) * gt[1] + (5.0f / 14.0f) * gt[2] + (5.0f / 28.0f) * gt[3] + (1.0f / 14.0f) * gt[4] + (1.0f / 56.0f) * gt[5];
			coeff = fabsf(coeff);
			if (coeff > maxCoeff)
				maxCoeff = coeff;
			//BC6
			coeff = gt[0] + 0.75f * gt[1] + (15.0f / 28.0f) * gt[2] + (5.0f / 14.0f) * gt[3] + (3.0f / 14.0f) * gt[4] + (3.0f / 28.0f) * gt[5] + (1.0f / 28.0f) * gt[6];
			coeff = fabsf(coeff);
			if (coeff > maxCoeff)
				maxCoeff = coeff;
			//BC7
			coeff = gt[0] + 0.875f * gt[1] + 0.75f * gt[2] + 0.625f * gt[3] + 0.5f * gt[4] + 0.375f * gt[5] + 0.25f * gt[6] + 0.125f * gt[7];
			coeff = fabsf(coeff);
			if (coeff > maxCoeff)
				maxCoeff = coeff;

			return maxCoeff;
		}
	};
}