#pragma once

#ifndef __NVCC__
#include <cassert>
#endif

#define HD __host__ __device__ __forceinline__
#define CONSTEXPR constexpr
#ifdef __INTELLISENSE__
#define debug_assert(x) while(false && (x)){};
#endif

#ifdef KERNEL_NO_DEBUG
#define debug_assert(...) (void(0))
#else
#define debug_assert(...) assert(__VA_ARGS__)
#endif

namespace kernel
{
	
	// Source: https://gist.github.com/thomcc/7739146

	template<typename A, typename B>
	struct pair
	{
		A first; B second;
	};

	namespace detail {

		// max base case
		template <class T>
		__host__ __device__ constexpr T const& do_max(T const& v) {
			return v;
		}

		// max inductive case
		template <class T, class... Rest> // requires SameType<T, Rest...>
		__host__ __device__ constexpr T const& do_max(T const& v0, T const& v1, Rest const &... rest) {
			return do_max(v0 < v1 ? v1 : v0, rest...);
		}

		// min base case
		template <class T>
		__host__ __device__ constexpr T const& do_min(T const& v) {
			return v;
		}

		// min variadic inductive case
		template <class T, class... Rest> // requires SameType<T, Rest...>
		__host__ __device__ constexpr T const& do_min(T const& v0, T const& v1, Rest const &...rest) {
			return do_min(v0 < v1 ? v0 : v1, rest...);
		}

		// min_max base case
		template <class T>
		__host__ __device__ constexpr pair<T const&, T const&>
			do_min_max(T const& cmin, T const& cmax) {
			return { cmin, cmax };
		}

		// min_max inductive case
		template <class T, class... Rest> // requires SameType<T, Rest...>
		__host__ __device__ constexpr pair<T const&, T const&>
			do_min_max(T const& cmin, T const& cmax, T const& next, Rest const &... rest) {
			return do_min_max(
				cmin < next ? cmin : next,
				next < cmax ? cmax : next,
				rest...
			);
		}

	} // namespace detail

	// public interface for minimum
	template <class T, class ...Rest> // requires SameType<T, Rest...>
	__host__ __device__ inline constexpr T const&
		minimum(T const& first, Rest const &... rest) {
		return detail::do_min(first, rest...);
	}

	// public interface for maximum
	template <class T, class ...Rest> // requires SameType<T, Rest...>
	__host__ __device__ inline constexpr T const&
		maximum(T const& first, Rest const &... rest) {
		return detail::do_max(first, rest...);
	}

	// public interface for min_max
	template <class T, class ...Rest> // requires SameType<T, Rest...>
	__host__ __device__ inline constexpr pair<T const&, T const&>
		min_max(T const& first, Rest const &... rest) {
		return detail::do_min_max(first, first, rest...);
	}
	
	// STRUCT TEMPLATE integral_constant
	template<class _Ty, _Ty _Val>
	struct integral_constant
	{	// convenient template for integral constant types
		enum
		{
			value = _Val
		};

		//using value_type = _Ty;
		//using type = integral_constant;
	};
	
}