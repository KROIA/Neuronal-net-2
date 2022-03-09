#pragma once
#include <stdio.h>
#include <string.h>
#include <vector>

#include "config.h"

class SignalVector;

class NET_API SignalVector
{
	public:
	SignalVector();
	SignalVector(size_t size);
	SignalVector(const SignalVector& other);
	SignalVector(const std::vector<float> & other);
	SignalVector(const float* begin, size_t elemCount);
	~SignalVector();

	const SignalVector& operator=(const SignalVector& other);
	float &operator[](size_t index) const;

	void resize(size_t size);
	void fill(const float* begin, size_t elemCount);

	size_t size() const;
	float* begin() const;
	float* end() const;

	void clear();

	private:
	size_t m_size;
	float* m_list;
};