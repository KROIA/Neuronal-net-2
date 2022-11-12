#pragma once
#include "unityConfig.h"
#include "backend/signalVector.h"

#ifdef MAKE_DLL
#define VEC_PARAM NeuronalNet::SignalVector* vec

UNITY_NET_API void* SignalVector_instantiate1();
UNITY_NET_API void* SignalVector_instantiate2(size_t size);
UNITY_NET_API void* SignalVector_instantiate3(const NeuronalNet::SignalVector* other);
UNITY_NET_API void* SignalVector_instantiate4(const std::vector<float>* other);
UNITY_NET_API void* SignalVector_instantiate5(const float* begin, size_t elemCount);
UNITY_NET_API void SignalVector_dealocate(VEC_PARAM);

UNITY_NET_API const NeuronalNet::SignalVector* SignalVector_assign(VEC_PARAM, const NeuronalNet::SignalVector* other);
UNITY_NET_API float* SignalVector_elementPtr(VEC_PARAM, size_t index);
UNITY_NET_API float SignalVector_getElement(VEC_PARAM, size_t index);
UNITY_NET_API void SignalVector_setElement(VEC_PARAM, size_t index, float value);

UNITY_NET_API void SignalVector_resize(VEC_PARAM, size_t size);
UNITY_NET_API void SignalVector_fill(VEC_PARAM, const float* begin, size_t elemCount);

UNITY_NET_API size_t SignalVector_size(VEC_PARAM);
UNITY_NET_API float* SignalVector_begin(VEC_PARAM);
UNITY_NET_API float* SignalVector_end(VEC_PARAM);

UNITY_NET_API void SignalVector_clear(VEC_PARAM);

UNITY_NET_API long double SignalVector_getSum(VEC_PARAM);
UNITY_NET_API float SignalVector_getMean(VEC_PARAM);				
UNITY_NET_API float SignalVector_getRootMeanSquare(VEC_PARAM);   
UNITY_NET_API float SignalVector_getGeometricMean(VEC_PARAM);    
UNITY_NET_API float SignalVector_getHarmonicMean(VEC_PARAM);		
#endif