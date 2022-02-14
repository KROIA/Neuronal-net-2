#pragma once

#include <iostream>
#include <stdio.h>
#include <string>
#include <chrono>

#include "config.h"

#ifdef UNIT_TEST
#include <vector>
#include <sstream>
#endif



namespace Debug
{

	class DebugFunctionTime;
	extern size_t __DBG_stackDepth;



	// helper functions for cleaner time measuring code
	extern NET_API std::chrono::time_point<std::chrono::high_resolution_clock> now();
	template <typename T>
	extern NET_API double milliseconds(T t);


	extern NET_API std::string timeToString(double timeMs);
	extern NET_API std::string bytesToString(size_t byteCount);

	class NET_API DebugFunctionTime
	{
		public:
		DebugFunctionTime(const std::string& funcName);
		~DebugFunctionTime();
	
		private:
		std::chrono::time_point<std::chrono::high_resolution_clock> t1;
		std::string m_stackSpace;
		std::string m_functionName;
	
	};

#ifdef UNIT_TEST
	extern NET_API std::vector<std::string> _unitTest_consoleBuffer;
#endif
};


#ifdef UNIT_TEST
#define STANDARD_CONSOLE_OUTPUT(x) \
{ \
	std::stringstream strm; \
	strm<<x; \
	Debug::_unitTest_consoleBuffer.push_back(strm.str()); \
}

#else
#define STANDARD_CONSOLE_OUTPUT(x) std::cout<<x;
#endif


#ifdef NET_DEBUG
#define CONSOLE_RAW(x) STANDARD_CONSOLE_OUTPUT(x);
#define CONSOLE_FUNCTION(x) STANDARD_CONSOLE_OUTPUT(std::string(Debug::__DBG_stackDepth*2, ' ')<<__FUNCTION__<<" : "<< x << "\n");
#define CONSOLE(x) STANDARD_CONSOLE_OUTPUT(std::string(Debug::__DBG_stackDepth*2, ' ')<< x << "\n");
#define DEBUG_FUNCTION_TIME_INTERVAL Debug::DebugFunctionTime ____DfuncTimeInt(__FUNCTION__);
#else
#define CONSOLE_RAW(x) ;
#define CONSOLE(x) ;
#define DEBUG_FUNCTION_TIME_INTERVAL ;
#endif


#define __VERIFY_RANGE_COMP1(min,var,max) if(min>var || var>max){ CONSOLE("Error: "<<#var<<" out of range: "<<min<<" > "<<#var<<" = "<<var<<" > "<<max)
#define VERIFY_RANGE(min,var,max) __VERIFY_RANGE_COMP1(min,var,max)}
#define VERIFY_RANGE(min,var,max,ret)__VERIFY_RANGE_COMP1(min,var,max) ret;}

#define __VERIFY_BOOL_COMP1(val,comp,message) if(val != comp){CONSOLE("Error: "<<message)
#define VERIFY_BOOL(val,comp,message) __VERIFY_BOOL_COMP1(val,comp,message)}
#define VERIFY_BOOL(val,comp,message,ret) __VERIFY_BOOL_COMP1(val,comp,message) ret;}

#define __VERIFY_VALID_PTR_COMP1(ptr, message) if(!ptr){CONSOLE("Error: "<<#ptr<<" == nullltr "<<message)
#define VERIFY_VALID_PTR(ptr, message) __VERIFY_VALID_PTR_COMP1(ptr,message)}
#define VERIFY_VALID_PTR(ptr, message, ret) __VERIFY_VALID_PTR_COMP1(ptr,message) ret;}