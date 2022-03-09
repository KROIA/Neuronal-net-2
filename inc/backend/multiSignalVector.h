#pragma once

#include "signalVector.h"

class MultiSignalVector;

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
	void fill(size_t vectorIndex, const SignalVector &vec);



	size_t size() const;
	size_t signalSize() const;
	const SignalVector** begin() const;
	const SignalVector** end() const;

	void clear();
	private:

	SignalVector** m_list;
	size_t m_vecCount;
	size_t m_signalCount;

};