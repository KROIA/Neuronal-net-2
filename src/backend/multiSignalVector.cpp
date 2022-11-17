#include "backend/multiSignalVector.h"

namespace NeuronalNet
{
	MultiSignalVector::MultiSignalVector()
        : m_grid(nullptr)
        , m_list(nullptr)
		, m_vecCount(0)
		, m_signalCount(0)
	{}
	MultiSignalVector::MultiSignalVector(const MultiSignalVector& other)
        : m_grid(nullptr)
        , m_list(nullptr)
	{
		m_vecCount = other.m_vecCount;
		m_signalCount = other.m_signalCount;
		m_list = DBG_NEW SignalVector * [m_vecCount];
		m_grid = DBG_NEW float* [m_vecCount];
		for (size_t i = 0; i < m_vecCount; ++i)
		{
			m_list[i] = DBG_NEW SignalVector(*other.m_list[i]);
			m_grid[i] = m_list[i]->begin();
		}
	}
	MultiSignalVector::MultiSignalVector(const std::vector<SignalVector>& other)
        : m_grid(nullptr)
        , m_list(nullptr)
	{
		m_vecCount = other.size();
		if (m_vecCount == 0)
		{
			m_signalCount = 0;
			return;
		}
		m_signalCount = other[0].size();
		m_list = DBG_NEW SignalVector * [m_vecCount];
		m_grid = DBG_NEW float* [m_vecCount];
		for (size_t i = 0; i < m_vecCount; ++i)
		{
			m_list[i] = DBG_NEW SignalVector(other[i]);
			m_grid[i] = m_list[i]->begin();
		}
	}
	MultiSignalVector::MultiSignalVector(const std::vector<std::vector<float> >& other)
        : m_grid(nullptr)
        , m_list(nullptr)
	{
		m_vecCount = other.size();
		if (m_vecCount == 0)
		{
			m_signalCount = 0;
			return;
		}
		m_signalCount = other[0].size();
		m_list = DBG_NEW SignalVector * [m_vecCount];
		m_grid = DBG_NEW float* [m_vecCount];
		for (size_t i = 0; i < m_vecCount; ++i)
		{
			m_list[i] = DBG_NEW SignalVector(other[i]);
			m_grid[i] = m_list[i]->begin();
		}
	}
	MultiSignalVector::MultiSignalVector(size_t vectorCount, size_t signalCount)
        : m_grid(nullptr)
        , m_list(nullptr)
	{
		m_vecCount = vectorCount;
		m_signalCount = signalCount;
		m_list = DBG_NEW SignalVector * [m_vecCount];
		m_grid = DBG_NEW float* [m_vecCount];
		for (size_t i = 0; i < m_vecCount; ++i)
		{
			m_list[i] = DBG_NEW SignalVector(m_signalCount);
			m_grid[i] = m_list[i]->begin();
		}
	}

	MultiSignalVector::~MultiSignalVector()
	{
		if (m_list)
		{
			for (size_t i = 0; i < m_vecCount; ++i)
			{
				delete m_list[i];
				m_list[i] = nullptr;
			}
			delete[] m_list;
			m_list = nullptr;
			delete[] m_grid;
			m_grid = nullptr;
			m_vecCount = 0;
			m_signalCount = 0;
		}
	}


	const MultiSignalVector& MultiSignalVector::operator=(const MultiSignalVector& other)
	{
		if (m_vecCount == other.m_vecCount)
		{
			for (size_t i = 0; i < m_vecCount; ++i)
			{
				m_list[i]->operator=(*other.m_list[i]);
				m_grid[i] = m_list[i]->begin();
			}

		}
		else
		{
			// Erase all 
			if (m_list)
			{
				for (size_t i = 0; i < m_vecCount; ++i)
				{
					delete m_list[i];
				}

				delete[] m_list;
				delete[] m_grid;
			}
			m_vecCount = other.m_vecCount;
			m_signalCount = other.m_signalCount;
			m_list = DBG_NEW SignalVector * [m_vecCount];
			m_grid = DBG_NEW float* [m_vecCount];
			for (size_t i = 0; i < m_vecCount; ++i)
			{
				m_list[i] = DBG_NEW SignalVector(*other.m_list[i]);
				m_grid[i] = m_list[i]->begin();
			}
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
        if (vectorCount == m_vecCount &&
            signalCount == m_signalCount)
			return;

		SignalVector** oldData = m_list;
		float** oldGrid		= m_grid;
		size_t oldVecCount = m_vecCount;
		size_t oldSigCount = m_signalCount;


        if(vectorCount == m_vecCount)
        {
            /*oldData = nullptr;
            oldGrid = nullptr;
            oldVecCount = 0;
            oldSigCount = 0;*/
        }
        else
        {
            m_list = DBG_NEW SignalVector * [vectorCount];
            m_grid = DBG_NEW float* [vectorCount];
        }
        m_vecCount = vectorCount;
        m_signalCount = signalCount;


		for (size_t i = 0; i < m_vecCount; ++i)
		{
			if (i < oldVecCount && oldSigCount == m_signalCount)
			{
                m_list[i] = oldData[i];
                //m_list[i] = DBG_NEW SignalVector(*oldData[i]);
                //m_grid[i] = m_list[i]->begin();
			}
			else
			{
				m_list[i] = DBG_NEW SignalVector(m_signalCount);
				m_grid[i] = m_list[i]->begin();
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
				m_grid[i] = m_list[i]->begin();
			}
		}

        if(oldData != m_list)
        {
            for (size_t i = 0; i < oldVecCount; ++i)
            {
                delete oldData[i];
            }
            delete oldData;
            delete oldGrid;
        }
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
		m_grid[vectorIndex] = m_list[vectorIndex]->begin();
	}
	void MultiSignalVector::fill(size_t vectorIndex, const SignalVector& vec)
	{
		if (vectorIndex >= m_vecCount)
			return;
		m_list[vectorIndex]->fill(vec.begin(), vec.size());
		m_grid[vectorIndex] = m_list[vectorIndex]->begin();
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
	const float ** MultiSignalVector::beginGrid() const
	{
		return (const float**)m_grid;
	}

	void MultiSignalVector::clear()
	{
		for (size_t i = 0; i < m_vecCount; ++i)
		{
			delete m_list[i];
		}
		delete[] m_list;
		delete[] m_grid;
		m_list = nullptr;
		m_grid = nullptr;
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
