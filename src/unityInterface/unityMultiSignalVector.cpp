#include "unityInterface/unityMultiSignalVector.h"

#ifdef MAKE_DLL
#define NULL_PTR_RETURN(ptr, returnVal) \
	if(!ptr) return returnVal;
#define NULL_PTR_RETURN_VOID(ptr) \
	if(!ptr) return;


using namespace NeuronalNet;

void* MultiSignalVector_instantiate1()
{
	return new MultiSignalVector();
}
void* MultiSignalVector_instantiate2(const NeuronalNet::MultiSignalVector * other)
{
	return new MultiSignalVector(*other);
}
void* MultiSignalVector_instantiate3(const std::vector<NeuronalNet::SignalVector>*other)
{
	return new MultiSignalVector(*other);
}
void* MultiSignalVector_instantiate4(const std::vector<std::vector<float> >*other)
{
	return new MultiSignalVector(*other);
}
void* MultiSignalVector_instantiate5(size_t vectorCount, size_t signalCount)
{
	return new MultiSignalVector(vectorCount, signalCount);
}
void MultiSignalVector_dealocate(VEC_PARAM)
{
	delete vec;
}


MultiSignalVector* MultiSignalVector_assign(VEC_PARAM, const MultiSignalVector * other)
{
	NULL_PTR_RETURN(vec, vec);
	vec->operator=(*other);
}
SignalVector* MultiSignalVector_getElementPtr(VEC_PARAM, size_t vectorIndex)
{
	NULL_PTR_RETURN(vec, nullptr);
	return &vec->operator[](vectorIndex);
}
void MultiSignalVector_setElement(VEC_PARAM, size_t vectorIndex, NeuronalNet::SignalVector* elem)
{
	NULL_PTR_RETURN_VOID(vec);
	vec->operator[](vectorIndex) = *elem;
}

void MultiSignalVector_resize1(VEC_PARAM, size_t vectorCount)
{
	NULL_PTR_RETURN(vec);
	vec->resize(vectorCount);
}
void MultiSignalVector_resize2(VEC_PARAM, size_t vectorCount, size_t signalCount)
{
	NULL_PTR_RETURN(vec);
	vec->resize(vectorCount, signalCount);
}
void MultiSignalVector_fill1(VEC_PARAM, const SignalVector **begin, size_t vecCount)
{
	NULL_PTR_RETURN(vec);
	vec->fill(begin, vecCount);
}
void MultiSignalVector_fill2(VEC_PARAM, const SignalVector * begin, size_t vecCount)
{
	NULL_PTR_RETURN(vec);
	vec->fill(begin, vecCount);
}
void MultiSignalVector_fill3(VEC_PARAM, size_t vectorIndex, const float* begin, size_t elemCount)
{
	NULL_PTR_RETURN(vec);
	vec->fill(vectorIndex, begin, elemCount);
}
void MultiSignalVector_fill4(VEC_PARAM, size_t vectorIndex, const SignalVector * fillWith)
{
	NULL_PTR_RETURN(vec);
	vec->fill(vectorIndex, *fillWith);
}



size_t MultiSignalVector_size(VEC_PARAM)
{
	NULL_PTR_RETURN(vec, 0);
	return vec->size();
}
size_t MultiSignalVector_signalSize(VEC_PARAM)
{
	NULL_PTR_RETURN(vec, 0);
	return vec->signalSize();
}
const SignalVector** MultiSignalVector_begin(VEC_PARAM)
{
	NULL_PTR_RETURN(vec, nullptr);
	return vec->begin();
}
const SignalVector** MultiSignalVector_end(VEC_PARAM)
{
	NULL_PTR_RETURN(vec, nullptr);
	return vec->end();
}

const float** MultiSignalVector_beginGrid(VEC_PARAM)
{
	NULL_PTR_RETURN(vec, nullptr);
	return vec->beginGrid();
}

void MultiSignalVector_clear(VEC_PARAM)
{
	NULL_PTR_RETURN(vec);
	vec->clear();
}


long double MultiSignalVector_getSum(VEC_PARAM)
{
	NULL_PTR_RETURN(vec, 0);
	return vec->getSum();
}
float MultiSignalVector_getMean(VEC_PARAM)
{
	NULL_PTR_RETURN(vec, 0);
	return vec->getMean();
}
float MultiSignalVector_getRootMeanSquare(VEC_PARAM)
{
	NULL_PTR_RETURN(vec, 0);
	return vec->getRootMeanSquare();
}
float MultiSignalVector_getGeometricMean(VEC_PARAM)
{
	NULL_PTR_RETURN(vec, 0);
	return vec->getGeometricMean();
}
float MultiSignalVector_getHarmonicMean(VEC_PARAM)
{
	NULL_PTR_RETURN(vec, 0);
	return vec->getHarmonicMean();
}


#endif