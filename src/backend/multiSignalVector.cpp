#include "backend/multiSignalVector.h"

namespace NeuronalNet
{
	MultiSignalVector::MultiSignalVector()
	{
		m_list = nullptr;
		m_vecCount = 0;
		m_signalCount = 0;
	}
	MultiSignalVector::MultiSignalVector(const MultiSignalVector& other)
	{
		m_vecCount = other.m_vecCount;
		m_signalCount = other.m_signalCount;
		m_list = new SignalVector * [m_vecCount];
		for (size_t i = 0; i < m_vecCount; ++i)
			m_list[i] = new SignalVector(*other.m_list[i]);
	}
	MultiSignalVector::MultiSignalVector(const std::vector<SignalVector>& other)
	{
		m_vecCount = other.size();
		if (m_vecCount == 0)
		{
			m_signalCount = 0;
			m_list = nullptr;
			return;
		}
		m_signalCount = other[0].size();
		m_list = new SignalVector * [m_vecCount];
		for (size_t i = 0; i < m_vecCount; ++i)
			m_list[i] = new SignalVector(other[i]);
	}
	MultiSignalVector::MultiSignalVector(const std::vector<std::vector<float> >& other)
	{
		m_vecCount = other.size();
		if (m_vecCount == 0)
		{
			m_signalCount = 0;
			m_list = nullptr;
			return;
		}
		m_signalCount = other[0].size();
		m_list = new SignalVector * [m_vecCount];
		for (size_t i = 0; i < m_vecCount; ++i)
			m_list[i] = new SignalVector(other[i]);
	}
	MultiSignalVector::MultiSignalVector(size_t vectorCount, size_t signalCount)
	{
		m_vecCount = vectorCount;
		m_signalCount = signalCount;
		m_list = new SignalVector * [m_vecCount];
		for (size_t i = 0; i < m_vecCount; ++i)
			m_list[i] = new SignalVector(m_signalCount);
	}

	MultiSignalVector::~MultiSignalVector()
	{
		if (m_list)
		{
			for (size_t i = 0; i < m_vecCount; ++i)
				delete m_list[i];
			delete[] m_list;
		}
	}


	const MultiSignalVector& MultiSignalVector::operator=(const MultiSignalVector& other)
	{
		if (m_vecCount == other.m_vecCount)
		{
			for (size_t i = 0; i < m_vecCount; ++i)
				m_list[i]->operator=(*other.m_list[i]);
		}
		else
		{
			// Erase all 
			if (m_list)
			{
				for (size_t i = 0; i < m_vecCount; ++i)
					delete m_list[i];
				delete[] m_list;
			}
			m_vecCount = other.m_vecCount;
			m_signalCount = other.m_signalCount;
			m_list = new SignalVector * [m_vecCount];
			for (size_t i = 0; i < m_vecCount; ++i)
				m_list[i] = new SignalVector(*other.m_list[i]);
		}
		return *this;
	}
	SignalVector& MultiSignalVector::operator[](size_t vectorIndex) const
	{
		return *m_list[vectorIndex];
	}


	void MultiSignalVector::resize(size_t vectorCount)
	{
		resize(vectorCount, m_signalCount);
	}
	void MultiSignalVector::resize(size_t vectorCount, size_t signalCount)
	{
		if (vectorCount == m_vecCount)
			return;

		SignalVector** oldData = m_list;
		size_t oldVecCount = m_vecCount;
		size_t oldSigCount = m_signalCount;

		m_vecCount = vectorCount;
		m_signalCount = signalCount;
		m_list = new SignalVector * [m_vecCount];
		for (size_t i = 0; i < m_vecCount; ++i)
		{
			if (i < oldVecCount && oldSigCount == m_signalCount)
			{
				m_list[i] = new SignalVector(*oldData[i]);
			}
			else
			{
				m_list[i] = new SignalVector(m_signalCount);
			}
		}
		if (oldSigCount != m_signalCount)
		{
			size_t loopCount = m_vecCount;
			if (loopCount > oldVecCount)
				loopCount = oldVecCount;
			for (size_t i = 0; i < loopCount; ++i)
			{
				m_list[i]->fill(oldData[i]->begin(), oldData[i]->size());
			}
		}

		for (size_t i = 0; i < oldVecCount; ++i)
		{
			delete oldData[i];
		}
		delete oldData;
	}
	void MultiSignalVector::fill(const SignalVector** begin, size_t vecCount)
	{
		for (size_t i = 0; i < vecCount; ++i)
			fill(i, (*begin[i]).begin(), (*begin[i]).size());
	}
	void MultiSignalVector::fill(const SignalVector* begin, size_t vecCount)
	{
		for (size_t i = 0; i < vecCount; ++i)
			fill(i, begin[i].begin(), begin[i].size());
	}
	void MultiSignalVector::fill(size_t vectorIndex, const float* begin, size_t elemCount)
	{
		if (vectorIndex >= m_vecCount)
			return;
		m_list[vectorIndex]->fill(begin, elemCount);
	}
	void MultiSignalVector::fill(size_t vectorIndex, const SignalVector& vec)
	{
		if (vectorIndex >= m_vecCount)
			return;
		m_list[vectorIndex]->fill(vec.begin(), vec.size());
	}



	size_t MultiSignalVector::size() const
	{
		return m_vecCount;
	}
	size_t MultiSignalVector::signalSize() const
	{
		return m_signalCount;
	}
	const SignalVector** MultiSignalVector::begin() const
	{
		return (const SignalVector**)m_list;
	}
	const SignalVector** MultiSignalVector::end() const
	{
		return (const SignalVector**)(m_list + m_vecCount);
	}

	void MultiSignalVector::clear()
	{
		for (size_t i = 0; i < m_vecCount; ++i)
			delete m_list[i];
		delete[] m_list;
		m_list = nullptr;
		m_vecCount = 0;
		m_signalCount = 0;
	}

	long double MultiSignalVector::getSum() const
	{
		long double sum = 0;
		for (size_t i = 0; i < m_vecCount; ++i)
			sum += m_list[i]->getSum();
		return sum;
	}
	float MultiSignalVector::getMean() const
	{
		if (m_vecCount == 0)
			return 0;
		long double sum = 0;
		size_t elemCount = 0;
		for (size_t i = 0; i < m_vecCount; ++i)
		{
			sum += m_list[i]->getSum();
			elemCount += m_list[i]->size();
		}
		return getSum() / (long double)elemCount;
	}
	float MultiSignalVector::getRootMeanSquare() const
	{
		if (m_vecCount == 0)
			return 0;
		long double sum = 0;
		size_t elemCount = 0;
		for (size_t i = 0; i < m_vecCount; ++i)
		{
			float* end = m_list[i]->begin() + m_list[i]->size();
			for (float* elem = m_list[i]->begin(); elem < end; ++elem)
			{
				sum += ((long double)(*elem) * (long double)(*elem));
			}
			elemCount += m_list[i]->size();
		}
		sum = sum / (long double)elemCount;
		return (float)sqrtl(sum);
	}
	float MultiSignalVector::getGeometricMean() const
	{
		if (m_vecCount == 0)
			return 0;
		long double product = 0;
		size_t elemCount = 0;
		for (size_t i = 0; i < m_vecCount; ++i)
		{
			float* end = m_list[i]->begin() + m_list[i]->size();
			for (float* elem = m_list[i]->begin(); elem < end; ++elem)
			{
				product *= (long double)(*elem);
			}
			elemCount += m_list[i]->size();
		}
		return (float)powl(product, (long double)1 / (long double)elemCount);
	}
	float MultiSignalVector::getHarmonicMean() const
	{
		if (m_vecCount == 0)
			return 0;
		long double sum = 0;
		size_t elemCount = 0;
		for (size_t i = 0; i < m_vecCount; ++i)
		{
			float* end = m_list[i]->begin() + m_list[i]->size();
			for (float* elem = m_list[i]->begin(); elem < end; ++elem)
			{
				sum += (long double)1 / (long double)(*elem);
			}
			elemCount += m_list[i]->size();
		}
		return (long double)elemCount / sum;
	}

};