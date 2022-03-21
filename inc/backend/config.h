#pragma once

//#define NET_DEBUG
//#define UNIT_TEST

// Use "_CrtDumpMemoryLeaks();" at the end of the application to dump the leaked memory
//#define NET_MEMORY_LEACK_CKECK


#ifdef DLL_EXPORT
#define NET_API __declspec(dllexport)
#else
#define NET_API __declspec(dllimport)
#endif

#if !defined(__PRETTY_FUNCTION__) && !defined(__GNUC__)
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif
