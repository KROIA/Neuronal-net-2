#pragma once

#include "backend/config.h"



#define VECTOR_INSERT_ONCE(vec,elem) \
	for (size_t i = 0; i < vec.size(); ++i) \
		if (vec[i] == elem) \
			goto endVectorInsert; \
	vec.push_back(elem); \
	endVectorInsert:; 

#define VECTOR_REMOVE_ELEM(vec,elem) \
	for (size_t i = 0; i < vec.size(); ++i) \
		if (vec[i] == elem) \
		{ \
			vec.erase(vec.begin() + i); \
			if (i > 0) \
				--i; \
		}

namespace NeuronalNet
{
	template<typename T>
	NET_API extern size_t getMaxIndex(const T* list, size_t size);
	template<typename T>
	NET_API extern size_t getMinIndex(const T* list, size_t size);
};