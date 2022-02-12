#include "..\inc\signalVector.h"

SignalVector::SignalVector()
{
	m_size = 0;
	m_list = nullptr;
}

SignalVector::SignalVector(size_t size)
{
	m_size = size;
	m_list = new float[m_size];
	memset(m_list, 0, m_size * sizeof(float));
}
SignalVector::SignalVector(const SignalVector& other)
{
	m_size = other.m_size;
	m_list = new float[m_size];
	memcpy(m_list, other.m_list, m_size * sizeof(float));
}

SignalVector::SignalVector(const std::vector<float>& other)
{
	m_size = other.size();
	m_list = new float[m_size];
	memcpy(m_list, other.data(), m_size * sizeof(float));
}
SignalVector::SignalVector(const float* begin, size_t elemCount)
{
	m_size = elemCount;
	m_list = new float[m_size];
	memcpy(m_list, begin, m_size * sizeof(float));
}

SignalVector::~SignalVector()
{
	if (m_list)
		delete[] m_list;
}

const SignalVector& SignalVector::operator=(const SignalVector& other)
{
	if (m_size == other.m_size)
	{
		memcpy(m_list, other.m_list, m_size * sizeof(float));
	}
	else
	{
		if (m_list)
			delete[] m_list;
		m_list = new float[other.m_size];
		m_size = other.m_size;
		memcpy(m_list, other.m_list, m_size * sizeof(float));
	}
	return *this;
}

float& SignalVector::operator[](size_t index) const
{
	return m_list[index];
}

void SignalVector::resize(size_t size)
{
	if (size == m_size)
		return;

	float* oldData = m_list;
	size_t oldSize = m_size;
	m_size = size;
	m_list = new float[m_size];
	
	size_t cpySize = m_size;
	if (cpySize > oldSize)
	{
		cpySize = oldSize;
		memset(m_list + oldSize, 0, (m_size - oldSize) * sizeof(float));
	}
	memcpy(m_list, oldData, cpySize*sizeof(float));
}
void SignalVector::fill(const float* begin, size_t elemCount)
{
	if (begin == nullptr)
		return;
	if (elemCount > m_size)
	{
		delete[] m_list;
		m_list = new float[elemCount];
		m_size = elemCount;
	}
	memcpy(m_list, begin, elemCount*sizeof(float));
}

size_t SignalVector::size() const
{
	return m_size;
}
float* SignalVector::begin() const
{
	return m_list;
}
float* SignalVector::end() const
{
	return m_list + m_size;
}