#include "renderer_math_cpu.h"
#include <Eigen/Eigenvalues>
#include <algorithm>

Eigen::MatrixXd RendererMathCpu::createCompanionMatrix(
	const Eigen::VectorXd& coefficients)
{
	Eigen::Index N = coefficients.size() - 1;
	Eigen::MatrixXd m = Eigen::MatrixXd::Zero(N, N);

	//diagonal, 1 below diagonal
	for (Eigen::Index i = 0; i < N - 1; ++i)
		m(i + 1, i) = 1;

	//add coefficients in the first row
	//scaled by 1/c[0], the highest degree
	for (Eigen::Index i = 0; i < N; ++i)
		m(0, i) = -coefficients[i + 1] / coefficients[0];

	return m;
}

Eigen::VectorXcd RendererMathCpu::rootsFromCompanionMatrix(const Eigen::MatrixXd& m)
{
	Eigen::EigenSolver<Eigen::MatrixXd> es(m, false);
	Eigen::VectorXcd eigenvalues = es.eigenvalues();
	return eigenvalues;
}

std::vector<double> RendererMathCpu::extractRealRoots(const Eigen::VectorXcd& complexRoots, double epsilon)
{
	std::vector<double> realRoots;
	realRoots.reserve(complexRoots.size());
	for (Eigen::Index i=0; i<complexRoots.size(); ++i)
	{
		if (std::abs(complexRoots[i].imag()) < epsilon)
			realRoots.push_back(complexRoots[i].real());
	}
	std::sort(realRoots.begin(), realRoots.end());
	return realRoots;
}
