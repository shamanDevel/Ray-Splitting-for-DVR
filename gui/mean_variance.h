#pragma once

/**
 * \brief Online algorithm to compute the mean and variance.
 * Source: http://datagenetics.com/blog/november22017/index.html
 */
class MeanVariance
{
public:
	typedef float real;

private:
	size_t n_;
	real mean_;
	real sn_;
	real lastmean_;

public:
	/**
	 * \brief Resets the running mean and variance.
	 */
	void reset()
	{
		n_ = 0;
		mean_ = 0;
		sn_ = 0;
		lastmean_ = 0;
	}
	MeanVariance() { reset(); }

	/**
	 * \brief Appends the new value 'x' to the running mean and variance computation.
	 */
	void append(real x)
	{
		n_ += 1;
		lastmean_ = mean_;
		mean_ += (x - lastmean_) / n_;
		if (n_ == 1)
			sn_ = 0;
		else
			sn_ += (x - lastmean_) * (x - mean_);
	}

	/**
	 * \brief Returns the current mean
	 */
	real mean() const { return mean_; }
	/**
	 * \brief Returns the current variance
	 */
	real var() const { return sn_ / n_; }
	/**
	 * \brief Returns the number of points
	 */
	size_t count() const { return n_; }
};