#pragma once

#include "backend/signalVector.h"

namespace NeuronalNet
{

	//class MultiSignalVector;

	class NET_API MultiSignalVector
	{
		public:
		MultiSignalVector();
		MultiSignalVector(const MultiSignalVector& other);
		MultiSignalVector(const std::vector<SignalVector>& other);
		MultiSignalVector(const std::vector<std::vector<float> >& other);
		MultiSignalVector(size_t vectorCount, size_t signalCount);
		~MultiSignalVector();

		const MultiSignalVector& operator=(const MultiSignalVector& other);
		SignalVector& operator[](size_t vectorIndex) const;

		void resize(size_t vectorCount);
		void resize(size_t vectorCount, size_t signalCount);
		void fill(const SignalVector** begin, size_t vecCount);
		void fill(const SignalVector* begin, size_t vecCount);
		void fill(size_t vectorIndex, const float* begin, size_t elemCount);
		void fill(size_t vectorIndex, const SignalVector& vec);



		size_t size() const;
		size_t signalSize() const;
		const SignalVector** begin() const;
		const SignalVector** end() const;

		const float ** beginGrid() const;

		void clear();


		long double getSum() const;			// SUM(elem)
		float getMean() const;				// SUM(elem)/nElem
		float getRootMeanSquare() const;    // sqrt(SUM(elem^2)/nElem)
		float getGeometricMean() const;     // pow(PROD(elem),1/nElem)
		float getHarmonicMean() const;		// nElem/SUM(1/elem)



		private:

		float** m_grid;
		SignalVector** m_list;
		size_t m_vecCount;
		size_t m_signalCount;

	};

};

