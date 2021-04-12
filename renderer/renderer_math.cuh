#pragma once

#include "helper_math.cuh"
#include "renderer_commons.cuh"

#ifndef __NVCC__
#include <iosfwd>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef FLT_MAX
#define FLT_MAX          3.402823466e+38F        // max value, from float.h
#endif

//Math library for polynomials

namespace kernel
{
	/**
	 * \brief 3D-lerping of the values at the voxel corners.
	 */
	HD float lerp3D(const float vals[8], const float3& p)
	{
		return lerp(
			lerp(lerp(vals[0], vals[1], p.x),
				lerp(vals[2], vals[3], p.x),
				p.y),
			lerp(lerp(vals[4], vals[5], p.x),
				lerp(vals[6], vals[7], p.x),
				p.y),
			p.z);
	}

	/**
	 * \brief computes \nabla lerp3D(vals, p)
	 */
	HD float3 lerp3DDerivatives(const float vals[8], const float3& p)
	{
		return make_float3(
			// d/dx
			lerp(
				lerp(vals[1]-vals[0], vals[3]-vals[2], p.y),
				lerp(vals[5]-vals[4], vals[7]-vals[6], p.y),
				p.z
			),
			// d/dy
			lerp(
				lerp(vals[2], vals[3], p.x) - lerp(vals[0], vals[1], p.x),
				lerp(vals[6], vals[7], p.x) - lerp(vals[4], vals[5], p.x),
				p.z
			),
			// d/dz
			lerp(lerp(vals[4], vals[5], p.x),
				lerp(vals[6], vals[7], p.x),
				p.y) - 
			lerp(lerp(vals[0], vals[1], p.x),
				lerp(vals[2], vals[3], p.x),
				p.y)
		);
	}

	template<typename T>
	HD void swap(T& a, T& b)
	{
		T c = a; a = b; b = c;
	}

	struct Combinatorics
	{
	public:
		static HD CONSTEXPR unsigned long long factorial(int n)
		{
			return n <= 1 ? 1 : (n * factorial(n - 1));
		}

	private:
		static HD CONSTEXPR unsigned long long binomial0(
			unsigned long long n, unsigned long long k,
			unsigned long long c, unsigned long long i)
		{
			return i > k
				? c
				: binomial0(n - 1, k, c * n / i, i + 1);
		}
	public:
		static HD CONSTEXPR unsigned long long binomial(unsigned long n, unsigned long k)
		{
			return binomial0(n, k > n - k ? n - k : k, 1, 1);
		}
	};

	/**
	 * \brief Defines various quadrature rules.
	 * All functions take a functor as first argument which must provide
	 * a method <code>Coeff_t operator()(Scalar_t x)</code> to evaluate
	 * the function at the given position.
	 */
	struct Quadrature
	{
		template<
			typename Functor_t,
			typename Scalar_t,
			typename Coeff_t = decltype(Functor_t()(Scalar_t()))>
			static HD Coeff_t rectangle(const Functor_t& f,
				const Scalar_t& a, const Scalar_t& b, const int N)
		{
			Scalar_t h = (b - a) / Scalar_t(N);
			Coeff_t ret = f(a);
			for (int i = 1; i < N; ++i)
				ret += f(a + i * h);
			return h * ret;
		}
		
		template<
			typename Functor_t,
			typename Scalar_t,
			typename Coeff_t = decltype(Functor_t()(Scalar_t()))>
		static HD Coeff_t trapezoid(const Functor_t& f,
			const Scalar_t& a, const Scalar_t& b, const int N)
		{
			//Novins 1992, Controlled Precision Volume Integration
			Scalar_t h = (b - a) / Scalar_t(N);
			Coeff_t ret = (f(a) + f(b)) / Scalar_t(2);
			for (int i = 1; i < N; ++i)
				ret += f(a + i * h);
			return h * ret;
		}

		template<
			typename Functor_t,
			typename Scalar_t,
			typename Coeff_t = decltype(Functor_t()(Scalar_t()))>
			static HD Coeff_t simpson(const Functor_t& f,
				const Scalar_t& a, const Scalar_t& b, const int N)
		{
			//Novins 1992, Controlled Precision Volume Integration
			debug_assert((N & 1) == 0); //N has to be even
			Scalar_t h = (b - a) / Scalar_t(N);
			Coeff_t ret = f(a) + f(b);
			for (int i = 1; i < N / 2; ++i)
				ret += 2 * f(a + 2 * i * h);
			for (int i = 1; i <= N / 2; ++i)
				ret += 4 * f(a + (2 * i - 1) * h);
			return h / Scalar_t(3) * ret;
		}

		/**
		 * \brief Adaptive Simpson scheme.
		 * The initial step size is given by 'h', and the final step size
		 * is also returned there.
		 * \tparam Functor_t 
		 * \tparam Scalar_t 
		 * \tparam Coeff_t 
		 * \param f the function to evaluate
		 * \param a the integration domain [a,b]
		 * \param b the integration domain [a,b]
		 * \param h in: initial step size, out: last step size
		 * \param epsilon the error bound that should be achieved
		 * \param N the number of performed evaluations
		 * \param hMin a minimal bound for the step size
		 * \param hMax a maximal bound for the step size
		 * \return the numerical approximation of the integral
		 */
		template<
			typename Functor_t,
			typename Scalar_t,
			typename Coeff_t = decltype(Functor_t()(Scalar_t()))>
			static HD Coeff_t adaptiveSimpson(const Functor_t& f,
				Scalar_t a, Scalar_t b, 
				Scalar_t& h, Scalar_t epsilon,
				int& N,
				const Scalar_t& hMin = 0, const Scalar_t& hMax = Scalar_t(FLT_MAX))
		{
			//Campagnolo 2015, Accurate Volume Rendering based on Adaptive Numerical Integration
			Coeff_t ret{ 0 };
			h = fmin(h, b - a);
			bool lastsuceeded = true;
			Coeff_t fx = f(a);
			Coeff_t fxh = f(a + h);
			N += 2;
			while (a < b)
			{
				//compute D and E
				Coeff_t fmid = f(a + h / 2);
				Coeff_t fq1 = f(a + h / 4);
				Coeff_t fq3 = f(a + h * 3 / 4);
				N += 3;
				Coeff_t S = h / Scalar_t(6) * (fx + 4 * fmid + fxh);
				Coeff_t D = h / Scalar_t(12) * (fx + 4 * fq1 + 2 * fmid + 4 * fq3 + fxh);
				Scalar_t E = 1 / Scalar_t(15) * maxCoeff(fabs(D - S));
				//adjust step size
				if (E <= epsilon || h<hMin)
				{
					//succeed
					ret += D;
					a += h;
					if (lastsuceeded && h<hMax)
					{
						h *= 2; //increase step
						epsilon *= 2;
					}
					else
						lastsuceeded = true;
					h = fmin(hMax, fmin(h, b - a));
					fx = fxh;
					fxh = f(a + h);
					N++;
				}
				else
				{
					//fail, retry with smaller step
					h /= 2;
					epsilon /= 2;
					lastsuceeded = false;
					fxh = fmid;
				}
			}
			return ret;
		}

		/**
		 * \brief Equal to \ref adaptiveSimpson,
		 * but terminates after the first successful step
		 * and returns the step size and integration value.
		 */
		template<
			typename Functor_t,
			typename Scalar_t,
			typename Coeff_t = decltype(Functor_t()(Scalar_t()))>
			static HD Coeff_t adaptiveSimpsonSingleStep(const Functor_t& f,
				Scalar_t a, Scalar_t b,
				Scalar_t& h, Scalar_t epsilon,
				Scalar_t& hNext, Scalar_t& epsilonNext,
				const Scalar_t& hMin = 0, const Scalar_t& hMax = Scalar_t(FLT_MAX))
		{
			//Campagnolo 2015, Accurate Volume Rendering based on Adaptive Numerical Integration
			h = fmin(h, b - a);
			bool lastsuceeded = true;
			Coeff_t fx = f(a);
			Coeff_t fxh = f(a + h);
			while (a < b)
			{
				//compute D and E
				Coeff_t fmid = f(a + h / 2);
				Coeff_t fq1 = f(a + h / 4);
				Coeff_t fq3 = f(a + h * 3 / 4);
				Coeff_t S = h / Scalar_t(6) * (fx + 4 * fmid + fxh);
				Coeff_t D = h / Scalar_t(12) * (fx + 4 * fq1 + 2 * fmid + 4 * fq3 + fxh);
				Coeff_t E = 1 / Scalar_t(15) * maxCoeff(fabs(D - S));
				//adjust step size
				if (E <= epsilon || h < hMin)
				{
					//succeed
					a += h;
					hNext = h;
					epsilonNext = epsilon;
					if (lastsuceeded && h < hMax)
					{
						hNext = h*2; //increase step
						epsilonNext = epsilon * 2;
					}
					else
						lastsuceeded = true;
					hNext = fmin(hMax, fmin(hNext, b - a));
					return D;
				}
				else
				{
					//fail, retry with smaller step
					h /= 2;
					epsilon /= 2;
					lastsuceeded = false;
					fxh = fmid;
				}
			}
			return { 0 };
		}
	};
	
    /**
     * \brief Array representation of Polynomial.
     * This class allows basic computer algebra to be performed with
     * Polynomial equations.
     *
     * For example, the polynomial \f$f(x) = 5 + 3x - 8x^2\f$ can be created like so:
     * \code{.cpp}
     * Polynomial<2, double> f{5, 3, -8};
     * \endcode
     * And evaluated at the point \f$x=3\f$ like so:
     * \code{.cpp}
     * double val = f(3);
     * \endcode
     *
     * Polynomial is an aggregate type, therefore, no constructors are defined.
     * Initialization works via initializer-lists.
     *
     * \tparam Order The order of the Polynomial.
     * \tparam Coeff_t The type of the coefficients of the Polynomial.
	 */
    template <int Order, class Coeff_t>
    struct Polynomial
    {
        static_assert(Order >= 0, "Order must be non-negative");

        using value_type = Coeff_t;
        using size_type = int;
        using difference_type = ptrdiff_t;
        using pointer = Coeff_t*;
        using const_pointer = const Coeff_t*;
        using reference = Coeff_t&;
        using const_reference = const Coeff_t&;
    	
        Coeff_t coeff[Order+1];

		//HD CONSTEXPR Polynomial() = default;
		//HD CONSTEXPR Polynomial(const MyType&) = default;
		//HD CONSTEXPR Polynomial(MyType&&) = default;
		//template <int OrderRhs>
		//HD CONSTEXPR Polynomial(const Polynomial<OrderRhs, Coeff_t>& p)
		//{
		//	static_assert(OrderRhs <= Order, "can only increase the order, not decrease");
		//	
		//}

        HD CONSTEXPR reference operator[](size_type _Pos) noexcept {
            return coeff[_Pos];
        }

        HD CONSTEXPR const_reference operator[](size_type _Pos) const noexcept {
            return coeff[_Pos];
        }
    	
    	HD CONSTEXPR int degree() const
        {
            return Order;
        }

        /**
         * \brief Optimized evaluation of the polynomial at x=0
         */
    	HD CONSTEXPR Coeff_t evalAtZero() const
    	{
            return coeff[0];
    	}
    	/**
    	 * \brief Optimized evaluation of the polynomial at x=1
    	 */
        HD CONSTEXPR Coeff_t evalAtOne() const
        {
            Coeff_t v = coeff[0];
            for (int i = 1; i <= Order; ++i)
                v += coeff[i];
            return v;
        }
    	/**
    	 * \brief Evaluates this polynomial at position 'x'.
    	 * It uses Horner's method.
    	 */
		template<typename Scalar_t>
    	HD CONSTEXPR Coeff_t operator()(const Scalar_t& x) const
    	{
            if (Order == 0) return coeff[0];
            Coeff_t sum = coeff[Order];
            for (int i = Order-1; i >= 0; --i)
                sum = sum * x + coeff[i];
            return sum;
    	}
        /**
         * \brief Evaluates this polynomial at position 'x'.
         * Delegates to operator()(x).
         */
		template<typename Scalar_t>
        HD CONSTEXPR Coeff_t eval(const Scalar_t& x) const { return operator()(x); }

    	/**
    	 * \brief Computes the derivative of this polynomial
    	 */
    	HD CONSTEXPR Polynomial<Order-1, Coeff_t> derivative() const
    	{
            static_assert(Order > 0, "The order of the polynomial must be >0");
            Polynomial<Order - 1, Coeff_t> ret;
            for (int i = 0; i < Order; ++i)
                ret[i] = coeff[i+1] * (i + 1);
            return ret;
    	}

		/**
		 * \brief Computes the derivative of this polynomial and transforms it
		 *  from the interval [0, tMax] to [0,1]
		 */
		template<typename Scalar_t>
		HD CONSTEXPR Polynomial<Order - 1, Coeff_t> derivativeAndTransformTo01(Scalar_t tMax) const
		{
			static_assert(Order > 0, "The order of the polynomial must be >0");
			Scalar_t facDyn = 1;
			Polynomial<Order - 1, Coeff_t> ret;
			for (int i = 0; i < Order; ++i) {
				ret[i] = coeff[i + 1] * (i + 1) * facDyn;
				facDyn *= tMax;
			}
			return ret;
		}

    	/**
    	 * \brief Computes the improper integral of this polynomial
    	 */
    	HD CONSTEXPR Polynomial<Order+1, Coeff_t> integrate() const
    	{
            Polynomial<Order + 1, Coeff_t> ret;
            ret[0] = Coeff_t(0);
            for (int i = 0; i <= Order; ++i)
                ret[i + 1] = coeff[i] / Coeff_t(i + 1);
            return ret;
    	}

		/**
		 * \brief Transforms this polynomial p(x) linearly into a polynomial p'(x)
		 * such that p'(xA)=dA, p'(xB)=dB.
		 *
		 * Let l(x, xA, dA, xB, dB) be a linear interpolation, i.e. l(x)=a+bx,
		 * such that l(xA)=dA, l(xB)=dB.
		 * Then this function returns
		 * l(p(x), p(xA), dA, p(xB), dB)
		 *
		 * Note that this function is undefined if p(xA)==p(xB)
		 *
		 * \param pxA: p(x entry)
		 * \param dA: target entry data
		 * \param pxB: p(x exit)
		 * \param dB: target exit data
		 */
		template<typename CoeffOut>
		HD CONSTEXPR Polynomial<Order, CoeffOut> lerp(
			const Coeff_t& pxA, const CoeffOut& dA, 
			const Coeff_t& pxB, const CoeffOut& dB) const
        {
			Polynomial<Order, CoeffOut> out;
			Coeff_t denom_tmp = (pxA - pxB);
			if (fabs(denom_tmp) < 1e-9) {
				//constant density. This can happen in cells of
				//constant density or if parallel to an isosurface+axis.
				return { dA };
			}
			Coeff_t denom = Coeff_t(1) / denom_tmp;
			CoeffOut num = dA - dB;
        	//handle zeroth coefficient
			out.coeff[0] = dA + num * (coeff[0] - pxA) * denom;
        	//handle all other
			for (int i = 1; i <= Order; ++i)
				out.coeff[i] = coeff[i] * num * denom;
			return out;
        }

		/**
		 * \brief Unitary negation operator
		 */
		HD CONSTEXPR Polynomial<Order, Coeff_t> operator-() const
        {
			Polynomial<Order, Coeff_t> out;
			for (int i = 0; i <= Order; ++i)
				out[i] = -coeff[i];
			return out;
        }
		
		/**
		 * \brief Returns this - rhs.
		 * Both polynomials must have the same type and number of coefficients.
		 */
		HD CONSTEXPR Polynomial<Order, Coeff_t> operator-(
			const Polynomial<Order, Coeff_t>& rhs) const
		{
			Polynomial<Order, Coeff_t> out;
			for (int i = 0; i <= Order; ++i)
				out[i] = coeff[i] - rhs[i];
			return out;
		}

		/**
		 * \brief Returns this - rhs where "rhs" is a scalar.
		 */
		HD CONSTEXPR Polynomial<Order, Coeff_t> operator-(
			const Coeff_t& rhs) const
		{
			Polynomial<Order, Coeff_t> out = *this;
			out[0] -= rhs;
			return out;
		}

		/**
		 * \brief Returns this + rhs.
		 * Both polynomials must have the same type.
		 */
		template<int RhsOrder>
		HD CONSTEXPR Polynomial<(Order>RhsOrder?Order:RhsOrder), Coeff_t> operator+(
			const Polynomial<RhsOrder, Coeff_t>& rhs) const
		{
			Polynomial<(Order > RhsOrder ? Order : RhsOrder), Coeff_t> out = {0};
			for (int i = 0; i <= Order; ++i)
				out[i] += coeff[i];
			for (int i = 0; i <= RhsOrder; ++i)
				out[i] += rhs[i];
			return out;
		}

		/**
		 * \brief Returns this + rhs.
		 * Both polynomials must have the same type.
		 */
		template<int RhsOrder>
		HD CONSTEXPR Polynomial<(Order > RhsOrder ? Order : RhsOrder), Coeff_t>& operator+=(
			const Polynomial<RhsOrder, Coeff_t>& rhs)
		{
			for (int i = 0; i <= RhsOrder; ++i)
				coeff[i] += rhs[i];
			return *this;
		}

		/**
		 * \brief Returns this + rhs where "rhs" is a scalar.
		 */
		HD CONSTEXPR Polynomial<Order, Coeff_t> operator+(
			const Coeff_t& rhs) const
		{
			Polynomial<Order, Coeff_t> out = *this;
			out[0] += rhs;
			return out;
		}

		/**
		 * \brief Performs polynomial multiplication of lhs and rhs.
		 */
		template<int M, typename RhsCoeff_t>
		friend HD CONSTEXPR Polynomial<Order+M, decltype(Coeff_t()* RhsCoeff_t())> operator*(
			Polynomial<Order, Coeff_t> lhs, const Polynomial<M, RhsCoeff_t>& rhs)
        {
			Polynomial<Order + M, decltype(Coeff_t()* RhsCoeff_t())> out{ 0 };
			for (int i = 0; i <= Order; ++i)
				for (int j = 0; j <= M; ++j)
					out[i + j] += lhs[i] * rhs[j];
        	
			return out;
        }

		/**
		 * \brief Performs polynomial multiplication of a polynomial and a scalar.
		 */
		friend HD CONSTEXPR Polynomial<Order, Coeff_t> operator*(
			Polynomial<Order, Coeff_t> poly, const Coeff_t& scalar)
		{
			Polynomial<Order, Coeff_t> out{ 0 };
			for (int i = 0; i <= Order; ++i)
				out[i] = poly[i] * scalar;
			return out;
		}

		/**
		 * \brief Performs polynomial multiplication of a polynomial and a scalar.
		 */
		friend HD CONSTEXPR Polynomial<Order, Coeff_t> operator*(
			const Coeff_t& scalar, Polynomial<Order, Coeff_t> poly)
		{
			Polynomial<Order, Coeff_t> out{ 0 };
			for (int i = 0; i <= Order; ++i)
				out[i] = poly[i] * scalar;
			return out;
		}

#ifndef __NVCC__
		__host__ friend std::ostream& operator<<(std::ostream& os, const Polynomial<Order, Coeff_t>& p)
        {
			os << p[0];
			for (int i = 1; i <= Order; ++i)
				if (i==1)
					os << "+" << p[i] << "*x";
				else
					os << "+" << p[i] << "*x^" << i;
			return os;
        }
#endif

        /**
		 * \brief Casts the coefficient type of this polynomial to a new type
		 *   using the specified casting functor.
		 * The casting functor must specify an operator()(Coeff_t s) const
		 * that transforms each individual coefficient of this polynomial.
		 * 
		 * \tparam CastingFunctor 
		 * \param functor the functor instance
		 * \return a polynomial with the coefficients transformed by the specified functor
		 *	(including type changes)
		 */
		template<typename CastingFunctor>
		HD CONSTEXPR
		Polynomial<Order, decltype(CastingFunctor()(Coeff_t()))>
		cast(const CastingFunctor& functor) const
		{
			Polynomial<Order, decltype(CastingFunctor()(Coeff_t()))> ret;
			for (int i = 0; i <= Order; ++i)
				ret[i] = functor(coeff[i]);
			return ret;
        }

        /**
		 * \brief Computes an upper bound of the polynomial in the interval [a,b].
		 * \tparam Scalar_t the scalar type of the evaluation
		 * \param a the lower interval bound
		 * \param b the upper interval bound
		 * \return an upper bound of the polynomial in [a,b]
		 */
		template<typename Scalar_t>
		HD CONSTEXPR
		Coeff_t upperBound(const Scalar_t& a, const Scalar_t& b) const
        {
			const Scalar_t c = fmax(fabs(a), fabs(b));
			Scalar_t cAccum = c;
			Coeff_t bound = fabs(coeff[0]);
        	for (int i=1;i<=Order; ++i)
        	{
				bound += fabs(coeff[i] * cAccum);
				cAccum *= c;
        	}
			return bound;
        }

        /**
    	 * \brief Linear transformation of this polynomial,
    	 *  i.e. it returns the polynomial g(x) given by g(x):=f(a+bx)
    	 *  where f is this polynomial.
    	 *
    	 * This is the opposite to \ref lerp() where the result
    	 * values are linearly transformed, whereas here the input
    	 * x is linearly transformed.
    	 *
    	 * Example usage: assume this polynomial is defined in the interval [a,b],
    	 * but we want to transform it to [0,1], e.g. to create a
    	 * Berstein representation.
    	 * Then <code>g(x) = f(a + x*(b-a)) = f.transform(a, b-a)</code>
    	 * gives the new polynomial g with <code>g(0)=f(a), g(1)=f(b)</code>.
    	 *  
    	 * \tparam Scalar_t the scalar data type
    	 * \param a the constant offset in f(a+bx)
    	 * \param b the linear scaling in f(a+bx)
    	 * \return the transformed polynomial
    	 */
    	template<typename Scalar_t>
		HD CONSTEXPR Polynomial<Order, Coeff_t> transform(
			Scalar_t a, Scalar_t b) const
        {
			//TODO: compile-time loop unrolling
        	//so that the binomial coefficients can be precomputed
        	
	        // create powers of a to Order
			Scalar_t ax[Order + 1];
			ax[0] = 1;
#pragma unroll
			for (int i = 0; i < Order; ++i) ax[i + 1] = ax[i] * a;

        	// fill in coefficients
			Polynomial<Order, Coeff_t> out{ 0 };
			Scalar_t bx = 1;
#pragma unroll
			for (int k=0; k<=Order; ++k)
			{
#pragma unroll
				for (int i=k; i<=Order; ++i)
				{
					out.coeff[k] += Combinatorics::binomial(i, k) * coeff[i] * ax[i - k] * bx;
				}
				bx *= b;
			}
			return out;
        }
    };

	//Fallback for operator* if it can't be deduced automatically (e.g. in CUDA)
	template<int N, int M, typename LhsCoeff_t, typename RhsCoeff_t>
	HD
	Polynomial<N+M, decltype(LhsCoeff_t() * RhsCoeff_t(0))>
	mul(const Polynomial<N, LhsCoeff_t>& lhs, const Polynomial<M, RhsCoeff_t>& rhs)
	{
		Polynomial<N + M, decltype(LhsCoeff_t()* RhsCoeff_t())> out{ 0 };
		for (int i = 0; i <= N; ++i)
			for (int j = 0; j <= M; ++j)
				out[i + j] += lhs[i] * rhs[j];
		return out;
	}


	/**
	 * \brief converts the float4 that stores the factors of a 3rd-degree polynomial
	 * from the voxel integration to a polynomial instance.
	 *
	 * The factors in 'poly' are stored with the highest degree first,
	 * as opposed to the Polynomial class
	 */
	template<typename Coeff_t>
	HD Polynomial<3, Coeff_t> float4ToPoly(const float4& poly)
	{
		return { Coeff_t(poly.w), Coeff_t(poly.z), Coeff_t(poly.y), Coeff_t(poly.x) };
	}


	/**
	 * \brief Representation of a polynomial in Berstein bounds.
	 * Berstein basis functions represent any N-order polynomial
	 * with N+1 basis functions in the interval [0,1].
	 *
	 * They are used here mainly to provide better bounds
	 * on the maximum in the interval [0,1]
	 * as would be possible with \ref Polynomial::upperBound(a,b).
	 * It uses the fact that if b_0, ..., b_N are the Berstein coefficients,
	 * then max{b_0, ..., b_N} is an upper bound of f(x) in [0,1].
	 *
	 * Bernstein is an aggregate type, therefore, no constructors are defined.
	 * Initialization works via initializer-lists.
	 * But usually, you want to create them from an polynomial using
	 * Berstein::fromPolynomial()
	 *
	 * \tparam Order The order of the Polynomial.
	 * \tparam Coeff_t The type of the coefficients of the Polynomial.
	 */
	template <int Order, class Coeff_t>
	struct Bernstein
	{
		Coeff_t coeff[Order + 1];

		/**
		 * \brief Creates the berstein basis functions from
		 * the specified polynomial defined in [0,1].
		 * \param p the polynomial
		 * \return the polynomial in Bernstein basis
		 */
		HD CONSTEXPR
		static Bernstein<Order, Coeff_t> fromPolynomial(
			const Polynomial<Order, Coeff_t> p)
		{
			Bernstein<Order, Coeff_t> out = { 0 };
#pragma unroll
			for (int k=0; k<=Order; ++k) for (int r=0; r<=k; ++r)
			{
				//TODO: recursive / template programming to force constexpr
				out.coeff[k] += p.coeff[r] *
					(Combinatorics::binomial(k, r) / float(Combinatorics::binomial(Order, r)));
			}
			return out;
		}

		/**
		 * \brief Evaluates this polynomial in Bernstein form
		 * at position x.
		 * \tparam Scalar_t the scalar type
		 * \param x the position to evaluate
		 * \return the value of the polynomial at x
		 */
		template<typename Scalar_t>
		HD CONSTEXPR
		Coeff_t operator()(const Scalar_t x) const
		{
			Scalar_t xx[Order + 1];
			xx[0] = Scalar_t(1);
			Scalar_t mxx[Order + 1];
			mxx[0] = Scalar_t(1);
#pragma unroll
			for (int i = 0; i < Order; ++i) {
				xx[i + 1] = xx[i] * x;
				mxx[i + 1] = mxx[i] * (1 - x);
			}

			Coeff_t out = { 0 };
#pragma unroll
			for (int k = 0; k <= Order; ++k)
				out += coeff[k] * Combinatorics::binomial(Order, k) * xx[k] * mxx[Order - k];

			return out;
		}

		/**
		 * \brief Evaluates this polynomial in Bernstein form
		 * at position x.
		 * \tparam Scalar_t the scalar type
		 * \param x the position to evaluate
		 * \return the value of the polynomial at x
		 */
		template<typename Scalar_t>
		HD CONSTEXPR
			Coeff_t eval(const Scalar_t x) const
		{
			return this->operator()(x);
		}

		/**
		 * \brief Returns an upper bound of the polynomial in [0,1]
		 */
		HD CONSTEXPR
		Coeff_t upperBound() const
		{
			Coeff_t m = coeff[0];
#pragma unroll
			for (int i = 1; i <= Order; ++i) 
				m = fmax(m, coeff[i]);
			return m;
		}

		/**
		 * \brief Returns a lower bound of the polynomial in [0,1]
		 */
		HD CONSTEXPR
		Coeff_t lowerBound() const
		{
			Coeff_t m = coeff[0];
#pragma unroll
			for (int i = 1; i <= Order; ++i)
				m = fmin(m, coeff[i]);
			return m;
		}

		/**
		 * \brief Returns an upper bound of abs(polynomial) in [0,1]
		 */
		HD CONSTEXPR
		Coeff_t absBound() const
		{
			Coeff_t m = fabs(coeff[0]);
#pragma unroll
			for (int i = 1; i <= Order; ++i)
				m = fmax(m, fabs(coeff[i]));
			return m;
		}
	};

	
	template<int OrderQ, typename TypeQ, int OrderP, typename TypeP>
	struct PolyExpPoly_D1;
	template<int OrderQ, typename TypeQ, int OrderP, typename TypeP>
	struct PolyExpPoly_D2;
	
	/**
	 * \brief Evaluates equations of the form f(x)=q(x)exp(p(x)) where q and p are polynomials.
	 * It provides utilities to evaluate f(x), and integrate f(x) in [a,b]
	 */
	template<int OrderQ, typename TypeQ, int OrderP, typename TypeP>
	struct PolyExpPoly
	{
		typedef Polynomial<OrderQ, TypeQ> PolyQ;
		typedef Polynomial<OrderP, TypeP> PolyP;
		typedef decltype(TypeQ() * TypeP()) Coeff_t;

		PolyQ q;
		PolyP p;

		CONSTEXPR PolyExpPoly() = default;
		HD CONSTEXPR PolyExpPoly(const PolyQ& q, const PolyP& p)
			: q(q), p(p) {}
		
		template<typename Scalar_t>
		HD CONSTEXPR Coeff_t operator()(const Scalar_t& x) const
		{
			return q(x) * exp(p(x));
		}
		/**
		 * \brief Evaluates this function at position 'x'.
		 * Delegates to operator()(x).
		 */
		template<typename Scalar_t>
		HD CONSTEXPR Coeff_t eval(const Scalar_t& x) const { return operator()(x); }

		template<typename Scalar_t>
		HD Coeff_t integrateRectangle(const Scalar_t& a, const Scalar_t& b, const int N) const
		{
			return Quadrature::rectangle(*this, a, b, N);
		}
		
		template<typename Scalar_t>
		HD Coeff_t integrateTrapezoid(const Scalar_t& a, const Scalar_t& b, const int N) const
		{
			return Quadrature::trapezoid(*this, a, b, N);
		}

		template<typename Scalar_t>
		HD Coeff_t integrateSimpson(const Scalar_t& a, const Scalar_t& b, const int N) const
		{
			return Quadrature::simpson(*this, a, b, N);
		}

		template<int N, typename Scalar_t>
		HD Coeff_t integratePowerSeries(const Scalar_t& a, const Scalar_t& b) const
		{
			static_assert(N > 0, "N must be greater zero");
			//Novins 1992, Controlled Precision Volume Integration
			TypeP A[N + 1];
			A[0] = exp(p[0]);
			Coeff_t In = q[0] * A[0] * (b - a);
			Scalar_t an = a, bn = b;
			for (int k=1; k<=N; ++k)
			{
				an *= a; bn *= b;
				
				TypeP accuA = 0;
				for (int j = 1; j <= min(k, OrderP); ++j)
					accuA += j * p[j] * A[k - j];
				A[k] = accuA / k;

				for (int j = 0; j <= min(k, OrderQ); ++j)
					In += q[j] * A[k - j] * (bn - an) / (k + 1);
			}
			return In;
		}

#ifndef __NVCC__
		template<typename Scalar_t>
		__host__ __noinline__ Coeff_t integratePowerSeries(
			const Scalar_t& a, const Scalar_t& b, int N) const
		{
			debug_assert(N > 0);
			//Novins 1992, Controlled Precision Volume Integration
			//noinline because of alloca
			TypeP* A = static_cast<TypeP*>(alloca(sizeof(TypeP) * (N + 1)));
			A[0] = exp(p[0]);
			Coeff_t In = q[0] * A[0] * (b - a);
			Scalar_t an = a, bn = b;
			for (int k = 1; k <= N; ++k)
			{
				an *= a; bn *= b;

				TypeP accuA = 0;
				for (int j = 1; j <= min(k, OrderP); ++j)
					accuA += j * p[j] * A[k - j];
				A[k] = accuA / k;

				for (int j = 0; j <= min(k, OrderQ); ++j)
					In += q[j] * A[k - j] * (bn - an) / (k + 1);
			}
			return In;
		}
#endif
		
		/**
		 * \brief Computes an upper bound of this term in [a,b].
		 * It uses simple polynomial bounds
		 */
		template<typename Scalar_t>
		HD CONSTEXPR Coeff_t upperBound(const Scalar_t& a, const Scalar_t& b) const
		{
			const auto boundP = p.upperBound(a, b);
			const auto boundQ = q.upperBound(a, b);
			return boundQ * exp(boundP);
		}

		/**
		 * \brief Computes an upper bound of this term in [a,b]
		 * using advanced Bernstein bounds
		 */
		template<typename Scalar_t>
		HD CONSTEXPR Coeff_t upperBoundBernstein(const Scalar_t& a, const Scalar_t& b) const
		{
			const auto boundP = Bernstein<OrderP, TypeP>::fromPolynomial(
				p.transform(a, b-a)).upperBound();
			const auto boundQ = Bernstein<OrderQ, TypeQ>::fromPolynomial(
				q.transform(a, b - a)).upperBound();
			return boundQ * exp(boundP);
		}

		/**
		 * \brief Computes an upper bound of abs(q epx(p)) in [a,b]
		 * using advanced Bernstein bounds
		 */
		template<typename Scalar_t>
		HD CONSTEXPR Coeff_t absBoundBernstein(const Scalar_t& a, const Scalar_t& b) const
		{
			const auto boundP = Bernstein<OrderP, TypeP>::fromPolynomial(
				p.transform(a, b - a)).absBound();
			const auto boundQ = Bernstein<OrderQ, TypeQ>::fromPolynomial(
				q.transform(a, b - a)).absBound();
			return boundQ * exp(boundP);
		}

		/**
		 * \brief Computes the first derivative of this expression
		 */
		HD CONSTEXPR PolyExpPoly_D1<OrderQ, TypeQ, OrderP, TypeP> d1() const;
		/**
		 * \brief Computes the second derivative of this expression
		 */
		HD CONSTEXPR PolyExpPoly_D2<OrderQ, TypeQ, OrderP, TypeP> d2() const;

		/**
		 * \brief Casts the coefficient type of this polynomial to a new type
		 *   using the specified casting functors.
		 * The casting functors must specify an operator()(Coeff_t s) const
		 * that transforms each individual coefficient of this polynomial.
		 *
		 * \tparam CastingQ the functor type to cast the polynomial q
		 * \tparam CastingP the functor type to cast the polynomial p
		 * \param functorQ the functor instance
		 * \param functorP the functor instance
		 * \return a PolyExpPoly with the coefficients transformed by the specified functor
		 *	(including type changes)
		 */
		template<typename CastingQ, typename CastingP>
		HD CONSTEXPR PolyExpPoly<
			OrderQ, decltype(CastingQ()(TypeQ())),
			OrderP, decltype(CastingP()(TypeP()))> cast(
				const CastingQ& functorQ = CastingQ(),
				const CastingP& functorP = CastingP()
			) const
		{
			return {
				q.cast(functorQ),
				p.cast(functorP)
			};
		}
		
	public:
		/**
		 * \brief Computes bounds on the N-th derivative
		 * using the recursive formula from Novins 1992.
		 *
		 * You have to pass bounds on each derivative
		 * starting with (0) -- the original polynomial --
		 * to the N-th derivative of q and p.
		 * 
		 * \tparam N the derivative
		 * \param boundsQ bounds on each derivative from 0 to N of q
		 * \param boundsP bounds on each derivative from 0 to N of p
		 * \return bound on the whole derivative
		 */
		template<int N>
		HD CONSTEXPR Coeff_t derivativeBoundsRecursive(
			TypeQ boundsQ[N+1], TypeP boundsP[N+1]) const
		{
			//create A array
			TypeP A[N + 1];
			A[0] = exp(boundsP[0]);
#pragma unroll
			for (int n=1; n<=N; ++n)
			{
				TypeP a = { 0 };
#pragma unroll
				for (int k = 1; k <= n; ++k)
					a += boundsP[k] * A[n - k] / Combinatorics::factorial(k - 1);
				A[n] = a / n;
			}
			
			//final bound
			Coeff_t out = { 0 };
#pragma unroll
			for (int j = 0; j <= N; ++j)
				out += boundsQ[j] * A[N - j] / Combinatorics::factorial(j);
			return Combinatorics::factorial(N) * out;
		}

	private:
		template<int N, int O, typename C, typename S>
		HD CONSTEXPR static void fillArraySimple(const Polynomial<O, C>& p, S a, S b, C bounds[], int i,
			kernel::integral_constant<int, N>)
		{
			static_assert(N > 0, "N must be >0");
			bounds[i] = p.upperBound(a, b);
			fillArraySimple(p.derivative(), a, b, bounds, i+1, kernel::integral_constant<int, N - 1>());
		}
		template<int O, typename C, typename S>
		HD CONSTEXPR static void fillArraySimple(const Polynomial<O, C>& p, S a, S b, C bounds[], int i,
			kernel::integral_constant<int, 0>)
		{
			bounds[i] = p.upperBound(a, b);
		}

		template<int N, int O, typename C, typename S>
		HD CONSTEXPR static void fillArrayBernstein(const Polynomial<O, C>& p, C bounds[], int i,
			const S& a, const S& b, kernel::integral_constant<int, N>)
		{
			static_assert(N > 0, "N must be >0");
			const auto pHat = p.transform(a, b);
			bounds[i] = kernel::Bernstein<O, C>::fromPolynomial(pHat).absBound();
			fillArrayBernstein(p.derivative(), bounds, i + 1, a,b, kernel::integral_constant<int, N - 1>());
		}
		template<int O, typename C, typename S>
		HD CONSTEXPR static void fillArrayBernstein(const Polynomial<O, C>& p, C bounds[], int i,
			const S& a, const S& b, kernel::integral_constant<int, 0>)
		{
			const auto pHat = p.transform(a, b);
			bounds[i] = kernel::Bernstein<O, C>::fromPolynomial(pHat).absBound();
		}
		
	public:
		/**
		 * \brief Computes bounds of q and p in the interval [a,b]
		 *  using simple polynomial bounds.
		 * Then it calls \ref derivativeBoundsRecursive.
		 * \tparam N the N-th derivative will be computed
		 * \param a the lower bound of the interval
		 * \param b the upper bound of the interval
		 * \return 
		 */
		template<int N, typename Scalar_t>
		HD CONSTEXPR Coeff_t derivativeBoundsSimple(Scalar_t a, Scalar_t b) const
		{
			TypeQ boundsQ[N + 1];
			fillArraySimple(q, a, b, boundsQ, 0, kernel::integral_constant<int, N>());
			TypeP boundsP[N + 1];
			fillArraySimple(p, a, b, boundsP, 0, kernel::integral_constant<int, N>());

			return derivativeBoundsRecursive<N>(boundsQ, boundsP);
		}

		/**
		 * \brief Computes bounds of q and p in the interval [a,b]
		 *  using simple polynomial bounds.
		 * Then it calls \ref derivativeBoundsRecursive.
		 * \tparam N the N-th derivative will be computed
		 * \param a the lower bound of the interval
		 * \param b the upper bound of the interval
		 * \return
		 */
		template<int N, typename Scalar_t>
		HD CONSTEXPR Coeff_t derivativeBoundsBernstein(Scalar_t a, Scalar_t b) const
		{
			//TODO: here is some error
			TypeQ boundsQ[N + 1];
			//const auto qhat = q.transform(a, b-a);
			fillArrayBernstein(q, boundsQ, 0, a,b, kernel::integral_constant<int, N>());
			TypeP boundsP[N + 1];
			//const auto phat = p.transform(a, b-a);
			fillArrayBernstein(p, boundsP, 0, a,b, kernel::integral_constant<int, N>());

			return derivativeBoundsRecursive<N>(boundsQ, boundsP);
		}
	};

	/**
	 * \brief Factory method for PolyExpPoly
	 */
	template<int OrderQ, typename TypeQ, int OrderP, typename TypeP>
	HD CONSTEXPR PolyExpPoly<OrderQ, TypeQ, OrderP, TypeP>
		polyExpPoly(const Polynomial<OrderQ, TypeQ>& p, const Polynomial<OrderP, TypeP>& q)
	{
		return PolyExpPoly<OrderQ, TypeQ, OrderP, TypeP>(p, q);
	}

	template<int OrderQ, typename TypeQ, int OrderP, typename TypeP>
	struct PolyExpPoly_D1
	{
		typedef Polynomial<OrderQ, TypeQ> PolyQ;
		typedef Polynomial<OrderQ-1, TypeQ> PolyQ_D1;
		typedef Polynomial<OrderP, TypeP> PolyP;
		typedef Polynomial<OrderP-1, TypeP> PolyP_D1;
		typedef decltype(TypeQ()* TypeP()) Coeff_t;

		PolyQ q;
		PolyP p;
		PolyQ_D1 q1;
		PolyP_D1 p1;
		
		CONSTEXPR PolyExpPoly_D1() = default;
		HD CONSTEXPR PolyExpPoly_D1(const PolyExpPoly<OrderQ, TypeQ, OrderP, TypeP>& pep)
			: q(pep.q), p(pep.p), q1(pep.q.derivative()), p1(pep.p.derivative()) {}

		template<typename Scalar_t>
		HD CONSTEXPR Coeff_t operator()(const Scalar_t & x) const
		{
			return exp(p(x)) * (q(x) * p1(x) + q1(x));
		}
		/**
		 * \brief Evaluates this function at position 'x'.
		 * Delegates to operator()(x).
		 */
		template<typename Scalar_t>
		HD CONSTEXPR Coeff_t eval(const Scalar_t & x) const { return operator()(x); }

		static constexpr int MaxExpandedOrderQ = OrderQ + (OrderP - 1);
		typedef PolyExpPoly<MaxExpandedOrderQ, Coeff_t, OrderP, TypeP> ExpandedPolyType;

		/**
		 * \brief expands the expressions here to a complete Poly-exp-poly instance.
		 */
		HD CONSTEXPR ExpandedPolyType expand() const
		{
			//auto newQ = q * p1 + q1
			// CUDA can't deduce the operator*
			auto newQ = mul<OrderQ, OrderP - 1, TypeQ, TypeP>(q, p1) + q1;
			return ExpandedPolyType(newQ, p);
		}
	};
	template <int OrderQ, typename TypeQ, int OrderP, typename TypeP>
	HD CONSTEXPR PolyExpPoly_D1<OrderQ, TypeQ, OrderP, TypeP> PolyExpPoly<OrderQ, TypeQ, OrderP, TypeP>::d1() const
	{
		return { *this };
	}

	template<int OrderQ, typename TypeQ, int OrderP, typename TypeP>
	struct PolyExpPoly_D2
	{
		typedef Polynomial<OrderQ, TypeQ> PolyQ;
		typedef Polynomial<OrderQ - 1, TypeQ> PolyQ_D1;
		typedef Polynomial<OrderQ - 2, TypeQ> PolyQ_D2;
		typedef Polynomial<OrderP, TypeP> PolyP;
		typedef Polynomial<OrderP - 1, TypeP> PolyP_D1;
		typedef Polynomial<OrderP - 2, TypeP> PolyP_D2;
		typedef decltype(TypeQ()* TypeP()) Coeff_t;

		PolyQ q;
		PolyP p;
		PolyQ_D1 q1;
		PolyP_D1 p1;
		PolyQ_D2 q2;
		PolyP_D2 p2;

		static constexpr int MaxExpandedOrderQ =
			(OrderQ + (OrderP - 1) + (OrderP - 1) > OrderQ + (OrderP - 2))
			? OrderQ + (OrderP - 1) + (OrderP - 1)
			: OrderQ + (OrderP - 2);
		typedef PolyExpPoly<MaxExpandedOrderQ, Coeff_t, OrderP, TypeP> ExpandedPolyType;
		
		CONSTEXPR PolyExpPoly_D2() = default;
		HD CONSTEXPR PolyExpPoly_D2(const PolyExpPoly<OrderQ, TypeQ, OrderP, TypeP>& pep)
			: q(pep.q), p(pep.p)
			, q1(q.derivative()), p1(p.derivative())
			, q2(q1.derivative()), p2(p1.derivative())
		{}

		template<typename Scalar_t>
		HD CONSTEXPR Coeff_t operator()(const Scalar_t& x) const
		{
			const auto p1x = p1(x);
			return exp(p(x)) * (
				q(x)*(p1x*p1x+p2(x)) + 2*q1(x)*p1x + q2(x));
		}
		/**
		 * \brief Evaluates this function at position 'x'.
		 * Delegates to operator()(x).
		 */
		template<typename Scalar_t>
		HD CONSTEXPR Coeff_t eval(const Scalar_t& x) const { return operator()(x); }

		/**
		 * \brief Computes an upper bound of this term in [a,b].
		 * Note that it gives tighter bounds if `expand().upperBound(a,b)` is used
		 * instead.
		 */
		template<typename Scalar_t>
		HD CONSTEXPR Coeff_t upperBound(const Scalar_t& a, const Scalar_t& b) const
		{
			const auto boundP1x = p1.upperBound(a, b);
			return exp(p.upperBound(a, b)) * (
				q.upperBound(a, b) * (boundP1x * boundP1x + p2.upperBound(a, b))
				+ 2 * q1.upperBound(a, b) * boundP1x + q2.upperBound(a, b));
		}

		/**
		 * \brief expands the expressions here to a complete Poly-exp-poly instance.
		 */
		HD CONSTEXPR ExpandedPolyType expand() const
		{
			//auto newQ = q * (p1 * p1 + p2) + 2 * q1 * p1 + q2;
			// CUDA can't deduce the operator*
			auto newQ = mul<OrderQ, 2*OrderP-2, TypeQ, TypeP>(q, (
					mul<OrderP-1, OrderP-1, TypeP, TypeP>(p1, p1) + p2))
				+ Coeff_t{ 2 } *mul<OrderQ - 1, OrderP - 1, TypeQ, TypeP>(q1, p1)
				+ q2;
			return ExpandedPolyType(newQ, p);
		}

	};
	template <int OrderQ, typename TypeQ, int OrderP, typename TypeP>
	HD CONSTEXPR PolyExpPoly_D2<OrderQ, TypeQ, OrderP, TypeP> PolyExpPoly<OrderQ, TypeQ, OrderP, TypeP>::d2() const
	{
		return { *this };
	}

	template<typename T>
	struct CubicPolynomialBase;

	template<>
	struct CubicPolynomialBase<float>
	{
		static constexpr float EPS1 = 1e-5;
		static constexpr float EPS2 = 1e-2;
		typedef float4 factor_t;
		static __host__ __device__ __inline__ float abs(float v) { return ::fabsf(v); }
		static __host__ __device__ __inline__ float sqrt(float v) { return ::sqrtf(v); }
		static __host__ __device__ __inline__ float cbrt(float v) { return ::cbrtf(v); }
		static __host__ __device__ __inline__ float min(float a, float b) { return ::fminf(a, b); }
		static __host__ __device__ __inline__ float max(float a, float b) { return ::fmaxf(a, b); }
		static __host__ __device__ __inline__ float sign(float v) { return ::fsignf(v); }
		static __host__ __device__ __inline__ float cosh(float v) { return ::coshf(v); }
		static __host__ __device__ __inline__ float acosh(float v) { return ::acoshf(v); }
		static __host__ __device__ __inline__ float cos(float v) { return ::cosf(v); }
		static __host__ __device__ __inline__ float sinh(float v) { return ::sinhf(v); }
		static __host__ __device__ __inline__ float asinh(float v) { return ::asinhf(v); }
		static __host__ __device__ __inline__ float sin(float v) { return ::sinf(v); }
	};
	template<>
	struct CubicPolynomialBase<double>
	{
		static constexpr double EPS1 = 1e-8;
		static constexpr double EPS2 = 1e-2; //used to distinguish the quadratic case, must be pretty large
		typedef double4 factor_t;
		static __host__ __device__ __inline__ double abs(double v) { return ::fabs(v); }
		static __host__ __device__ __inline__ double sqrt(double v) { return ::sqrt(v); }
		static __host__ __device__ __inline__ double cbrt(double v) { return ::cbrt(v); }
		static __host__ __device__ __inline__ double min(double a, double b) { return ::fmin(a, b); }
		static __host__ __device__ __inline__ double max(double a, double b) { return ::fmax(a, b); }
		static __host__ __device__ __inline__ double sign(double v) { return ::fsign(v); }
		static __host__ __device__ __inline__ double cosh(double v) { return ::cosh(v); }
		static __host__ __device__ __inline__ double acosh(double v) { return ::acosh(v); }
		static __host__ __device__ __inline__ double cos(double v) { return ::cos(v); }
		static __host__ __device__ __inline__ double sinh(double v) { return ::sinh(v); }
		static __host__ __device__ __inline__ double asinh(double v) { return ::asinh(v); }
		static __host__ __device__ __inline__ double sin(double v) { return ::sin(v); }
	};

	/**
	 * \brief Utilities to handle cubic polynomials.
	 *
	 * A cubic polynomial is defined as f(x)=ax^3+bx^2+cx+d.
	 * The factors a,b,c,d are stored as float4 (a=x, b=y, c=z, d=w)
	 */
	template<typename T>
	struct CubicPolynomial : CubicPolynomialBase<T>
	{
		using typename CubicPolynomialBase<T>::factor_t;
		using CubicPolynomialBase<T>::EPS1;
		using CubicPolynomialBase<T>::EPS2;

		using CubicPolynomialBase<T>::abs;
		using CubicPolynomialBase<T>::sqrt;
		using CubicPolynomialBase<T>::cbrt;
		using CubicPolynomialBase<T>::min;
		using CubicPolynomialBase<T>::max;
		using CubicPolynomialBase<T>::sign;
		using CubicPolynomialBase<T>::cosh;
		using CubicPolynomialBase<T>::acosh;
		using CubicPolynomialBase<T>::cos;
		using CubicPolynomialBase<T>::sinh;
		using CubicPolynomialBase<T>::asinh;
		using CubicPolynomialBase<T>::sin;

		static __host__ __device__ __inline__ T safeSqrt(T v)
		{
			return sqrt(max(T(0), v));
		}
		static __host__ __device__ __inline__ T safeAcos(T v)
		{
			return acos(max(T(-1), min(T(1), v)));
		}
		static __host__ __device__ __inline__ bool isZero(float v)
		{
			return v > -EPS1 && v < EPS1;
		}

		/**
		 * Returns the cubic polynomial factors for the ray r(t)=entry+t*dir
		 * traversing the voxel with corner values given by 'vals' and accessing the
		 * tri-linear interpolated values.
		 */
		static __host__ __device__ factor_t getFactors(
			const float vals[8], const float3& entry, const float3& dir)
		{
			const T v0 = vals[0], v1 = vals[1], v2 = vals[2], v3 = vals[3];
			const T v4 = vals[4], v5 = vals[5], v6 = vals[6], v7 = vals[7];
			const T ex = entry.x, ey = entry.y, ez = entry.z;
			const T dx = dir.x, dy = dir.y, dz = dir.z;

#define GET_FACTOR_VERSION 3
#if GET_FACTOR_VERSION==1
			// expanded version
			const T a = -dx * dy * dz * v0 + dx * dy * dz * v1 + dx * dy * dz * v2 - dx * dy * dz * v3 + dx * dy * dz * v4 - dx * dy * dz * v5 - dx * dy * dz * v6 + dx * dy * dz * v7;
			const T b = -dx * dy * ez * v0 + dx * dy * ez * v1 + dx * dy * ez * v2 - dx * dy * ez * v3 + dx * dy * ez * v4 - dx * dy * ez * v5 - dx * dy * ez * v6 + dx * dy * ez * v7 + dx * dy * v0 - dx * dy * v1 - dx * dy * v2 + dx * dy * v3 - dx * dz * ey * v0 + dx * dz * ey * v1 + dx * dz * ey * v2 - dx * dz * ey * v3 + dx * dz * ey * v4 - dx * dz * ey * v5 - dx * dz * ey * v6 + dx * dz * ey * v7 + dx * dz * v0 - dx * dz * v1 - dx * dz * v4 + dx * dz * v5 - dy * dz * ex * v0 + dy * dz * ex * v1 + dy * dz * ex * v2 - dy * dz * ex * v3 + dy * dz * ex * v4 - dy * dz * ex * v5 - dy * dz * ex * v6 + dy * dz * ex * v7 + dy * dz * v0 - dy * dz * v2 - dy * dz * v4 + dy * dz * v6;
			const T c = -dx * ey * ez * v0 + dx * ey * ez * v1 + dx * ey * ez * v2 - dx * ey * ez * v3 + dx * ey * ez * v4 - dx * ey * ez * v5 - dx * ey * ez * v6 + dx * ey * ez * v7 + dx * ey * v0 - dx * ey * v1 - dx * ey * v2 + dx * ey * v3 + dx * ez * v0 - dx * ez * v1 - dx * ez * v4 + dx * ez * v5 - dx * v0 + dx * v1 - dy * ex * ez * v0 + dy * ex * ez * v1 + dy * ex * ez * v2 - dy * ex * ez * v3 + dy * ex * ez * v4 - dy * ex * ez * v5 - dy * ex * ez * v6 + dy * ex * ez * v7 + dy * ex * v0 - dy * ex * v1 - dy * ex * v2 + dy * ex * v3 + dy * ez * v0 - dy * ez * v2 - dy * ez * v4 + dy * ez * v6 - dy * v0 + dy * v2 - dz * ex * ey * v0 + dz * ex * ey * v1 + dz * ex * ey * v2 - dz * ex * ey * v3 + dz * ex * ey * v4 - dz * ex * ey * v5 - dz * ex * ey * v6 + dz * ex * ey * v7 + dz * ex * v0 - dz * ex * v1 - dz * ex * v4 + dz * ex * v5 + dz * ey * v0 - dz * ey * v2 - dz * ey * v4 + dz * ey * v6 - dz * v0 + dz * v4;
			const T d = -ex * ey * ez * v0 + ex * ey * ez * v1 + ex * ey * ez * v2 - ex * ey * ez * v3 + ex * ey * ez * v4 - ex * ey * ez * v5 - ex * ey * ez * v6 + ex * ey * ez * v7 + ex * ey * v0 - ex * ey * v1 - ex * ey * v2 + ex * ey * v3 + ex * ez * v0 - ex * ez * v1 - ex * ez * v4 + ex * ez * v5 - ex * v0 + ex * v1 + ey * ez * v0 - ey * ez * v2 - ey * ez * v4 + ey * ez * v6 - ey * v0 + ey * v2 - ez * v0 + ez * v4 + v0;
#elif GET_FACTOR_VERSION==2
			// factored version
			// a bit faster, but more numerically unstable !?!
			const T t1 = -v0 + v1 + v2 - v3 + v4 - v5 - v6 + v7;
			const T t2 = v0 - v1 - v2 + v3;
			const T t3 = v0 - v1 - v4 + v5;
			const T t4 = v0 - v2 - v4 + v6;
			const T a = (dx * dy * dz) * t1;
			const T b = (dx * dy * ez) * t1 + (dx * dy) * t2 + (dx * dz * ey) * t1 + (dx * dz) * t3 + (dy * dz * ex) * t1 + (dy * dz) * t4;
			const T c = (dx * ey * ez) * t1 + (dx * ey) * t2 + (dx * ez) * t3 + dx * (-v0 + v1) + (dy * ex * ez) * t1 + (dy * ex) * t2 + (dy * ez) * t4 + dy * (-v0 + v2) + (dz * ex * ey) * t1 + (dz * ex) * t3 + (dz * ey) * t4 + dz * (-v0 + v4);
			const T d = (ex * ey * ez) * t1 + (ex * ey) * t2 + (ex * ez) * t3 + ex * (-v0 + v1) + (ey * ez) * t4 + ey * (-v0 + v2) + ez * (-v0 + v4) + v0;
#elif GET_FACTOR_VERSION==3
			//Based on "Interactive ray tracing for isosurface rendering"
			//by Steven Parker et al., 1998
			
			//reorder values, z first
			const T values[8] = { v0, v4, v2, v6, v1, v5, v3, v7 };
			//assemble basis functions
			const T uA[2] = { 1 - ex, ex };
			const T vA[2] = { 1 - ey, ey };
			const T wA[2] = { 1 - ez, ez };
			const T uB[2] = { -dx, dx };
			const T vB[2] = { -dy, dy };
			const T wB[2] = { -dz, dz };
			//compute factors
			T a = 0;
			T b = 0;
			T c = 0;
			T d = 0; // -isovalue;
			int valueIndex = 0;
			for (int i = 0; i < 2; ++i) {
				for (int j = 0; j < 2; ++j) {
					for (int k = 0; k < 2; ++k) {
						a += uB[i] * vB[j] * wB[k] * values[valueIndex];
						b += (uA[i] * vB[j] * wB[k] + uB[i] * vA[j] * wB[k] + uB[i] * vB[j] * wA[k]) * values[valueIndex];
						c += (uB[i] * vA[j] * wA[k] + uA[i] * vB[j] * wA[k] + uA[i] * vA[j] * wB[k]) * values[valueIndex];
						d += uA[i] * vA[j] * wA[k] * values[valueIndex];
						valueIndex++;
					}
				}
			}
#endif
			return factor_t{ a, b, c, d };
		}

		static __host__ __device__ T evalCubic(const factor_t& factors, T t)
		{
			return factors.w + t * (factors.z + t * (factors.y + t * factors.x));
		}

		/**
		 * \brief Computes the roots of the cubic using the analytic hyperbolic equations.
		 * \param factors the factors of the polynomial f(x)=ax^3+bx^2+cx+d = 0.
		 * \param roots will be filled with the values of 'x' at the roots
		 * \return the number of real roots, 0, 1, 3
		 */
		static __host__ __device__ int rootsHyperbolic(const factor_t& factors, T roots[3])
		{
			//extract factors
			const T a = factors.x, b = factors.y, c = factors.z, d = factors.w;

			if (abs(a) <= EPS2)
			{
				if (isZero(b))
				{
					//linear equation
					if (isZero(c)) return 0; //constant
					roots[0] = -d / c;
					return 1;
				}
				//quadratic equation
				T discr = c * c - T(4) * b * d;
				if (discr < 0) return 0;
				if (isZero(discr))
				{
					roots[0] = -c / (T(2) * b);
					return 1;
				}
				else {
					discr = sqrt(discr);
					//https://math.stackexchange.com/questions/866331/numerically-stable-algorithm-for-solving-the-quadratic-equation-when-a-is-very
					roots[0] = (-c - sign(c) * discr) / (2 * b);
					roots[1] = d / (b * roots[0]);
					//roots[0] = (-c + discr) / (T(2) * b);
					//roots[1] = (-c - discr) / (T(2) * b);
					return 2;
				}
			}

			//convert to depressed cubic t^3+pt+q=0
			const T p = (T(3) * a * c - b * b) / (T(3) * a * a);
			const T q = (T(2) * b * b * b - T(9) * a * b * c + T(27) * a * a * d) / (T(27) * a * a * a);

#define t2x(t) ((t)-b/(3*a))

			if (abs(p) <= EPS1)
			{
				//there exists exactly one root
				roots[0] = t2x(cbrt(-q));
				return 1;
			}
			//formular of Francois Viète
			//https://en.wikipedia.org/wiki/Cubic_equation#Trigonometric_solution_for_three_real_roots
			const T Delta = T(4) * p * p * p + T(27) * q * q;
			if (Delta > 0)
			{
				//one real root
				T t0;
				if (p < 0)
					t0 = T(-2) * sign(q) * sqrt(-p / T(3)) * cosh(T(1) / T(3) * acosh(T(-3) * abs(q) / (T(2) * p) * sqrt(T(-3) / p)));
				else
					t0 = T(-2) * sqrt(p / T(3)) * sinh(T(1) / T(3) * asinh(T(3) * q / (T(2) * p) * sqrt(T(3) / p)));
				roots[0] = t2x(t0);
				return 1;
			}
			//TODO: handle double root if Delta>-EPS1:
			// simple root at 3q/p
			// double root at -3a/2p
			else
			{
				//three real roots
				const T f1 = T(2) * safeSqrt(-p / T(3));
				const T f2 = T(1) / T(3) * safeAcos(T(3) * q / (T(2) * p) * safeSqrt(-T(3) / p));
				for (int k = 0; k < 3; ++k)
					roots[k] = t2x(f1 * cos(f2 - T(2) * T(M_PI) * k / T(3)));
				return 3;
			}

#undef t2x
		}

		/**
		 * \brief Computes the roots of the cubic using the analytic Schwarze's equations.
		 * Source: http://www.realtimerendering.com/resources/GraphicsGems/gems/Roots3And4.c
		 *
		 * \param factors the factors of the polynomial f(x)=ax^3+bx^2+cx+d = 0.
		 * \param roots will be filled with the values of 'x' at the roots
		 * \return the number of real roots, 0, 1, 3
		 */
		static __host__ __device__ int rootsSchwarze(const factor_t& factors, T roots[3])
		{
			int     i, num;
			double  sub;
			double  A, B, C;
			double  sq_A, p, q;
			double  cb_p, D;

			/* normal form: x^3 + Ax^2 + Bx + C = 0 */

			A = factors.y / factors.x;
			B = factors.z / factors.x;
			C = factors.w / factors.x;

			/*  substitute x = y - A/3 to eliminate quadric term:
			x^3 +px + q = 0 */

			sq_A = A * A;
			p = 1.0 / 3 * (-1.0 / 3 * sq_A + B);
			q = 1.0 / 2 * (2.0 / 27 * A * sq_A - 1.0 / 3 * A * B + C);

			/* use Cardano's formula */

			cb_p = p * p * p;
			D = q * q + cb_p;

#define     EQN_EPS     1e-9
#define	    IsZero(x)	((x) > -EQN_EPS && (x) < EQN_EPS)
			
			if (IsZero(D))
			{
				if (IsZero(q)) /* one triple solution */
				{
					roots[0] = 0;
					num = 1;
				}
				else /* one single and one double solution */
				{
					double u = cbrt(-q);
					roots[0] = 2 * u;
					roots[1] = -u;
					num = 2;
				}
			}
			else if (D < 0) /* Casus irreducibilis: three real solutions */
			{
				double phi = 1.0 / 3 * acos(-q / sqrt(-cb_p));
				double t = 2 * sqrt(-p);

				roots[0] = t * cos(phi);
				roots[1] = -t * cos(phi + M_PI / 3);
				roots[2] = -t * cos(phi - M_PI / 3);
				num = 3;
			}
			else /* one real solution */
			{
				double sqrt_D = sqrt(D);
				double u = cbrt(sqrt_D - q);
				double v = -cbrt(sqrt_D + q);

				roots[0] = u + v;
				num = 1;
			}

#undef IsZero
#undef EQN_EPS

			/* resubstitute */

			sub = 1.0 / 3 * A;

			for (i = 0; i < num; ++i)
				roots[i] -= sub;

			return num;
		}

	};

	/**
	 * \brief Implements Marmitt's algorithm to find the first root in an interval [t0, t1]
	 * of a cubic polynomial.
	 * \tparam NumIterations the number of iterations to refine the root, default=3
	 * \tparam StableQuadratic if a more stable quadratic formula should be used, default=true
	 */
	template<int NumIterations = 3,
		bool StableQuadratic = true>
		struct Marmitt
	{
	private:
		static __host__ __device__ __inline__ float neubauer(
			float t0, float t1, float v0, float v1,
			const float vals[8], const float3& entry, const float3& dir, float isovalue)
		{
			for (int i = 0; i < NumIterations; ++i)
			{
#ifndef NEUBAUER_USE_LINEAR
#define NEUBAUER_USE_LINEAR 0
#endif

#if NEUBAUER_USE_LINEAR==1
				float t = t0 + (t1 - t0) * v0 / (v0 - v1);
#else
				float t = 0.5 * (t0 + t1);
#endif
				float v = lerp3D(vals, entry + t * dir) - isovalue;
				if (isign(v) == isign(v0))
				{
					t0 = t;
					v0 = v;
				}
				else
				{
					t1 = t;
					v1 = v;
				}
			}
			return t0 + (t1 - t0) * v0 / (v0 - v1);
		}
		
	public:
		/**
		 * \brief Computes the first intersection of the ray and the tri-linear interpolated isosurface.
		 * It uses first the cubic polynomial defined by 'factors', computed
		 * via CubicPolynomial<ScalarType>::getFactors(vals, entry, dir) to determine the candidates.
		 * But then it uses lerp3D to refine the intersection, as this is more stable
		 * than the polynomial (but a bit more expensive).
		 * \param vals the values at the corners of the voxel
		 * \param factors the factors of the cubic equation, see CubicPolynomial<ScalarType>::getFactors(vals, entry, dir)
		 * \param isovalue the isovalue to intersect against
		 * \param entry the entry location
		 * \param dir the ray direction, already normalized by the voxel size (originalDir/voxelSize)
		 * \param tEntry the time of entry
		 * \param tExit the time of exit.
		 * \param timeOut the time of the first intersection in [tEntry, tExit], changed only if there is an intersection
		 * \return true if there is an intersection
		 */
		template<typename factor_t>
		static __host__ __device__ __inline__ bool eval(
			const float vals[8], const factor_t& factors, const float isovalue,
			const float3& entry, const float3& dir, float tEntry, float tExit,
			float& timeOut)
		{
			float t0 = tEntry, t1 = tExit;
			float v0 = lerp3D(vals, entry + t0 * dir) - isovalue;
			float v1 = lerp3D(vals, entry + t1 * dir) - isovalue;

			//find extrema of cubic equation / roots of derivative
			//f'(t) = At^2+2Bt+C
			const auto A = 3 * factors.x, B = 2 * factors.y, C = factors.z;
			auto discr = B * B - 4 * A * C;
			if (discr > 0)
			{
				//we have extrema
				discr = sqrt(discr);
				float e0, e1;
				if (!StableQuadratic)
				{
					//simple quadratic formula
					e0 = (-B - discr) / (2 * A);
					e1 = (-B + discr) / (2 * A);
				}
				else
				{
					//more numerically stable:
					//https://math.stackexchange.com/questions/866331/numerically-stable-algorithm-for-solving-the-quadratic-equation-when-a-is-very
					e0 = (-B - fsignf(B) * discr) / (2 * A);
					e1 = C / (A * e0);
				}
				if (e0 > e1) swap(e0, e1);
				if (e0 >= t0 && e0 <= t1)
				{
					float ve0 = lerp3D(vals, entry + e0 * dir) - isovalue;
					if (isign(ve0) == isign(v0))
					{
						//advance the ray to the second segment
						t0 = e0; v0 = ve0;
					}
					else
					{
						//stay in the first segment
						t1 = e0; v1 = ve0;
					}
				}
				if (e1 >= t0 && e1 <= t1)
				{
					float ve1 = lerp3D(vals, entry + e1 * dir) - isovalue;
					if (isign(ve1) == isign(v0))
					{
						//advance the ray to the third segment
						t0 = e1; v0 = ve1;
					}
					else
					{
						t1 = e1; v1 = ve1;
					}
				}
			}
			if (isign(v0) == isign(v1)) return false; //no hit

			//now we know we've got a root in t0,t1
			//find it via repeated linear interpolation
			timeOut = neubauer(t0, t1, v0, v1, vals, entry, dir, isovalue);
			return true;
		}



		/**
		 * \brief Computes the first intersection of the ray and the tri-linear interpolated isosurface.
		 * It uses first the cubic polynomial defined by 'factors', computed
		 * via CubicPolynomial<ScalarType>::getFactors(vals, entry, dir) to determine the candidates.
		 * But then it uses lerp3D to refine the intersection, as this is more stable
		 * than the polynomial (but a bit more expensive).
		 * \param vals the values at the corners of the voxel
		 * \param factors the factors of the cubic equation, see CubicPolynomial<ScalarType>::getFactors(vals, entry, dir)
		 * \param isovalue the isovalue to intersect against
		 * \param entry the entry location
		 * \param dir the ray direction, already normalized by the voxel size (originalDir/voxelSize)
		 * \param tEntry the time of entry
		 * \param tExit the time of exit.
		 * \param timesOut the times of intersection in [tEntry, tExit]
		 * \return the number of intersections found
		 */
		template<typename factor_t>
		static __host__ __device__ __inline__ int evalAll(
			const float vals[8], const factor_t& factors, const float isovalue,
			const float3& entry, const float3& dir, float tEntry, float tExit,
			float timesOut[3])
		{
			int numRoot = 0;
			float t0 = tEntry, t1 = tExit;
			float v0 = lerp3D(vals, entry + t0 * dir) - isovalue;
			float v1 = lerp3D(vals, entry + t1 * dir) - isovalue;

			//find extrema of cubic equation / roots of derivative
			//f'(t) = At^2+2Bt+C
			const auto A = 3 * factors.x, B = 2 * factors.y, C = factors.z;
			auto discr = B * B - 4 * A * C;
			if (discr > 0)
			{
				//we have extrema
				discr = sqrt(discr);
				float e0, e1;
				if (!StableQuadratic)
				{
					//simple quadratic formula
					e0 = (-B - discr) / (2 * A);
					e1 = (-B + discr) / (2 * A);
				}
				else
				{
					//more numerically stable:
					//https://math.stackexchange.com/questions/866331/numerically-stable-algorithm-for-solving-the-quadratic-equation-when-a-is-very
					e0 = (-B - fsignf(B) * discr) / (2 * A);
					e1 = C / (A * e0);
				}
				if (e0 > e1) swap(e0, e1);
				float ve0 = lerp3D(vals, entry + e0 * dir) - isovalue;
				float ve1 = lerp3D(vals, entry + e1 * dir) - isovalue;

				if (e0 >= t0 && e0 <= t1)
				{
					if (isign(ve0) != isign(v0))
					{
						//evaluate intersection in the first segment
						timesOut[numRoot] = neubauer(t0, e0, v0, ve0, vals, entry, dir, isovalue);
						numRoot++;
					}
					//advance the ray to the second segment
					t0 = e0; v0 = ve0;
				}
				if (e1 >= t0 && e1 <= t1)
				{
					if (isign(ve1) != isign(v0))
					{
						//evaluate intersection in the first segment
						timesOut[numRoot] = neubauer(t0, e1, v0, ve1, vals, entry, dir, isovalue);
						//if (numRoot > 0) extrema[0] = e0;
						numRoot++;
					}
					//advance the ray to the third segment
					t0 = e1; v0 = ve1;
				}
			}
			if (isign(v0) != isign(v1))
			{
				timesOut[numRoot] = neubauer(t0, t1, v0, v1, vals, entry, dir, isovalue);
				//if (numRoot > 0) extrema[numRoot - 1] = t0;
				numRoot++;
			}
			return numRoot;
		}
	};
}