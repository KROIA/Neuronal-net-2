#pragma once
#include "backend/config.h"

#ifdef MAKE_DLL
#ifdef DLL_EXPORT
#define UNITY_NET_API extern "C" NET_API
#else
#define UNITY_NET_API
#endif
#endif