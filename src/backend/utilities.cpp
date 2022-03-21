#include "backend/utilities.h"


namespace NeuronalNet
{
	template<typename T>
	size_t getMaxIndex(const T* begin, size_t size)
	{
		const T* end = begin + size;
		size_t maxIt = 0;
		for (const T* it = begin; it < end; ++it)
		{
			if (*it > *(begin + maxIt))
				maxIt = it - begin;
		}
		return maxIt;
	}
	template NET_API size_t getMaxIndex<float>(const float* list, size_t size);
	template NET_API size_t getMaxIndex<double>(const double* list, size_t size);
	template NET_API size_t getMaxIndex<int>(const int* list, size_t size);
	template NET_API size_t getMaxIndex<unsigned int>(const unsigned int* list, size_t size);


	template<typename T>
	size_t getMinIndex(const T* begin, size_t size)
	{
		const T* end = begin + size;
		size_t minIt = 0;
		for (const T* it = begin; it < end; ++it)
		{
			if (*it < *(begin + minIt))
				minIt = it - begin;
		}
		return minIt;
	}
	template NET_API size_t getMinIndex<float>(const float* list, size_t size);
	template NET_API size_t getMinIndex<double>(const double* list, size_t size);
	template NET_API size_t getMinIndex<int>(const int* list, size_t size);
	template NET_API size_t getMinIndex<unsigned int>(const unsigned int* list, size_t size);


}; 

