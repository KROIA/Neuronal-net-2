#include "unityInterface/unitySignalVector.h"

#ifdef MAKE_DLL
#define NULL_PTR_RETURN(ptr, returnVal) \
	if(!ptr) return returnVal;
#define NULL_PTR_RETURN_VOID(ptr) \
	if(!ptr) return;


using namespace NeuronalNet;

void* SignalVector_instantiate1()
{
	return new SignalVector();
}
void* SignalVector_instantiate2(size_t size)
{
	return new SignalVector(size);
}
void* SignalVector_instantiate3(const NeuronalNet::SignalVector* other)
{
	return new SignalVector(*other);
}
void* SignalVector_instantiate4(const std::vector<float>* other)
{
	return new SignalVector(*other);
}
void* SignalVector_instantiate5(const float* begin, size_t elemCount)
{
	return new SignalVector(begin, elemCount);
}
void SignalVector_dealocate(VEC_PARAM)
{
	delete vec;
}

const NeuronalNet::SignalVector* SignalVector_assign(VEC_PARAM, const NeuronalNet::SignalVector* other)
{
	NULL_PTR_RETURN(vec, vec);
	return &vec->operator=(*other);
}
float* SignalVector_elementPtr(VEC_PARAM, size_t index)
{
	NULL_PTR_RETURN(vec, nullptr);
	return &vec->operator[](index);
}
float SignalVector_getElement(VEC_PARAM, size_t index)
{
	NULL_PTR_RETURN(vec, 0);
	return vec->operator[](index);
}
void SignalVector_setElement(VEC_PARAM, size_t index, float value)
{
	NULL_PTR_RETURN_VOID(vec);
	vec->operator[](index) = value;
}

void SignalVector_resize(VEC_PARAM, size_t size)
{
	NULL_PTR_RETURN_VOID(vec);
	vec->resize(size);
}
void SignalVector_fill(VEC_PARAM, const float* begin, size_t elemCount)
{
	NULL_PTR_RETURN_VOID(vec);
	vec->fill(begin, elemCount);
}

size_t SignalVector_size(VEC_PARAM)
{
	NULL_PTR_RETURN(vec, 0);
	return vec->size();
}
float* SignalVector_begin(VEC_PARAM)
{
	NULL_PTR_RETURN(vec, nullptr);
	return vec->begin();
}
float* SignalVector_end(VEC_PARAM)
{
	NULL_PTR_RETURN(vec, nullptr);
	return vec->end();
}

void SignalVector_clear(VEC_PARAM)
{
	NULL_PTR_RETURN_VOID(vec);
	vec->clear();
}

long double SignalVector_getSum(VEC_PARAM)
{
	NULL_PTR_RETURN(vec, 0);
	return vec->getSum();
}
float SignalVector_getMean(VEC_PARAM)
{
	NULL_PTR_RETURN(vec, 0);
	return vec->getMean();
}
float SignalVector_getRootMeanSquare(VEC_PARAM)
{
	NULL_PTR_RETURN(vec, 0);
	return vec->getRootMeanSquare();
}
float SignalVector_getGeometricMean(VEC_PARAM)
{
	NULL_PTR_RETURN(vec, 0);
	return vec->getGeometricMean();
}
float SignalVector_getHarmonicMean(VEC_PARAM)
{
	NULL_PTR_RETURN(vec, 0);
	return vec->getHarmonicMean();
}

#endif