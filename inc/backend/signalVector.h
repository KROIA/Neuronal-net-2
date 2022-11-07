#pragma once
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#include "backend/config.h"
#include "backend/debug.h"

namespace NeuronalNet
{
	//class SignalVector;

	class NET_API SignalVector
	{
		public:
		SignalVector();
		SignalVector(size_t size);
		SignalVector(const SignalVector& other);
		SignalVector(const std::vector<float>& other);
		SignalVector(const float* begin, size_t elemCount);
		~SignalVector();

		const SignalVector& operator=(const SignalVector& other);
		float& operator[](size_t index) const;

		void resize(size_t size);
		void fill(const float* begin, size_t elemCount);

		size_t size() const;
		float* begin() const;
		float* end() const;

		void clear();

		long double getSum() const;			// SUM(elem)
		float getMean() const;				// SUM(elem)/nElem
		float getRootMeanSquare() const;    // sqrt(SUM(elem^2)/nElem)
		float getGeometricMean() const;     // pow(PROD(elem),1/nElem)
		float getHarmonicMean() const;		// nElem/SUM(1/elem)

		private:
		size_t m_size;
		float* m_list;
	};

};
