#pragma once

#include <vector>
#include <complex>
#include <Eigen/Core>

#include "renderer_math.cuh"

/**
 * Additional math operations executed on the CPU.
 */
class RendererMathCpu
{

private:
	static Eigen::MatrixXd createCompanionMatrix(const Eigen::VectorXd& coefficients);
	static Eigen::VectorXcd rootsFromCompanionMatrix(const Eigen::MatrixXd& m);

public:
	/**
	 * \brief Computes all roots of the given polynomial using
	 * the eigenvalues of the companion matrix.
	 * \param poly the polynomial of degree N
	 * \tparam N the order of the polynomial
	 * \tparam Coeff_t the coefficient type, float or double
	 * \return the vector of roots
	 */
	template<size_t N, typename Coeff_t>
	static Eigen::VectorXcd roots(const kernel::Polynomial<N, Coeff_t>& poly)
	{
		//Matlab scheme: highest degree first
		Eigen::VectorXd coefficients(N + 1);
		for (size_t i = 0; i <= N; ++i)
			coefficients[N - i] = static_cast<double>(poly.coeff[i]);
		return rootsFromCompanionMatrix(createCompanionMatrix(coefficients));
	}

	/**
	 * Extracts the list of real-valued roots from the complex roots.
	 * \param complexRoots the complex roots
	 * \param epsilon the threshold to determine if the imaginary part is small enough to
	 * deem that root to be real-valued.
	 * \return a list of real-valued roots, sorted ascending by value
	 */
	static std::vector<double> extractRealRoots(
		const Eigen::VectorXcd& complexRoots, double epsilon = 1e-7f);
};
