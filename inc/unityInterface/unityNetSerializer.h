#pragma once
#include "unityConfig.h"
#include "backend/netSerializer.h"

#ifdef MAKE_DLL
#define SERIALIZER_PARAM NeuronalNet::NetSerializer* serializer

UNITY_NET_API void* NetSerializer_instantiate();
UNITY_NET_API void NetSerializer_dealocate(SERIALIZER_PARAM);

UNITY_NET_API void NetSerializer_setFilePath(SERIALIZER_PARAM, const char* path);
UNITY_NET_API const char* NetSerializer_getFilePath(SERIALIZER_PARAM);

UNITY_NET_API bool NetSerializer_saveToFile1(SERIALIZER_PARAM, NeuronalNet::Net* net);
UNITY_NET_API bool NetSerializer_saveToFile2(SERIALIZER_PARAM, NeuronalNet::GeneticNet* net);

UNITY_NET_API bool NetSerializer_readFromFile1(SERIALIZER_PARAM, NeuronalNet::Net* net);
UNITY_NET_API bool NetSerializer_readFromFile2(SERIALIZER_PARAM, NeuronalNet::GeneticNet* net);
#endif