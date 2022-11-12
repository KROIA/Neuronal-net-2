#pragma once
#include "unityConfig.h"
#include "backend/multiSignalVector.h"

#ifdef MAKE_DLL
#define VEC_PARAM NeuronalNet::MultiSignalVector* vec

UNITY_NET_API void* MultiSignalVector_instantiate1();
UNITY_NET_API void* MultiSignalVector_instantiate2(const NeuronalNet::MultiSignalVector* other);
UNITY_NET_API void* MultiSignalVector_instantiate3(const std::vector<NeuronalNet::SignalVector>* other);
UNITY_NET_API void* MultiSignalVector_instantiate4(const std::vector<std::vector<float> >* other);
UNITY_NET_API void* MultiSignalVector_instantiate5(size_t vectorCount, size_t signalCount);
UNITY_NET_API void MultiSignalVector_dealocate(VEC_PARAM);

UNITY_NET_API NeuronalNet::MultiSignalVector* MultiSignalVector_assign(VEC_PARAM, const NeuronalNet::MultiSignalVector* other);
UNITY_NET_API NeuronalNet::SignalVector* MultiSignalVector_getElementPtr(VEC_PARAM, size_t vectorIndex);
UNITY_NET_API void MultiSignalVector_setElement(VEC_PARAM, size_t vectorIndex, NeuronalNet::SignalVector *elem);


UNITY_NET_API void MultiSignalVector_resize1(VEC_PARAM, size_t vectorCount);
UNITY_NET_API void MultiSignalVector_resize2(VEC_PARAM, size_t vectorCount, size_t signalCount);
UNITY_NET_API void MultiSignalVector_fill1(VEC_PARAM, const NeuronalNet::SignalVector** begin, size_t vecCount);
UNITY_NET_API void MultiSignalVector_fill2(VEC_PARAM, const NeuronalNet::SignalVector* begin, size_t vecCount);
UNITY_NET_API void MultiSignalVector_fill3(VEC_PARAM, size_t vectorIndex, const float* begin, size_t elemCount);
UNITY_NET_API void MultiSignalVector_fill4(VEC_PARAM, size_t vectorIndex, const NeuronalNet::SignalVector* fillWith);



UNITY_NET_API size_t MultiSignalVector_size(VEC_PARAM);
UNITY_NET_API size_t MultiSignalVector_signalSize(VEC_PARAM);
UNITY_NET_API const NeuronalNet::SignalVector** MultiSignalVector_begin(VEC_PARAM);
UNITY_NET_API const NeuronalNet::SignalVector** MultiSignalVector_end(VEC_PARAM);

UNITY_NET_API const float** MultiSignalVector_beginGrid(VEC_PARAM);

UNITY_NET_API void MultiSignalVector_clear(VEC_PARAM);


UNITY_NET_API long double MultiSignalVector_getSum(VEC_PARAM);		
UNITY_NET_API float MultiSignalVector_getMean(VEC_PARAM);				
UNITY_NET_API float MultiSignalVector_getRootMeanSquare(VEC_PARAM);   
UNITY_NET_API float MultiSignalVector_getGeometricMean(VEC_PARAM);    
UNITY_NET_API float MultiSignalVector_getHarmonicMean(VEC_PARAM);		

#endif