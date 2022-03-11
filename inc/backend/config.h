#pragma once


//#define NET_DEBUG
//#define UNIT_TEST


#ifdef DLL_EXPORT
#define NET_API __declspec(dllexport)
#else
#define NET_API __declspec(dllimport)
#endif

#if !defined(__PRETTY_FUNCTION__) && !defined(__GNUC__)
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif
