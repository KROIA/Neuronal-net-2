/*#include "..\inc\signalVector.h"

SignalVector::SignalVector()
{
	m_size = 0;
	m_list = nullptr;
}

SignalVector::SignalVector(size_t size)
{
	m_size = size;
	m_list = new float[m_size];
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

	}
	else
	{
		if (m_list)
			delete[] m_list;

	}
}

float& SignalVector::operator[](size_t index)
{
	// TODO: hier return-Anweisung eingeben
}

size_t SignalVector::size() const
{
	return size_t();
}
*/