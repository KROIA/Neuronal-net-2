#pragma once
#define _CRTDBG_MAP_ALLOC
#include <cstdlib>
#include <crtdbg.h>

#include <iostream>
#include <stdio.h>
#include <string>
#include <chrono>

#include "config.h"

#ifdef UNIT_TEST
#include <vector>
#include <sstream>
#endif


#if  defined(_DEBUG) && defined(NET_MEMORY_LEACK_CKECK)
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
// Replace _NORMAL_BLOCK with _CLIENT_BLOCK if you want the
// allocations to be of _CLIENT_BLOCK type
#else
#define DBG_NEW new
#endif

namespace NeuronalNet
{
	namespace Debug
	{

		class DebugFunctionTime;
		extern size_t __DBG_stackDepth;

		extern NET_API std::string timeToString(double timeMs);
		extern NET_API std::string bytesToString(size_t byteCount);

		class NET_API Timer
		{
			public:
			Timer(bool autoStart = false);
			~Timer();

			void start();
			void stop();
			void pause();
			void unpause();
			double getMillis() const;
			void reset();
			bool isRunning() const;
			bool isPaused() const;

			static inline std::chrono::time_point<std::chrono::high_resolution_clock> getCurrentTimePoint();
			template <typename T>
			static inline double getMillis(T t);

			protected:
			std::chrono::time_point<std::chrono::high_resolution_clock> t1;
			std::chrono::time_point<std::chrono::high_resolution_clock> t2;

			std::chrono::time_point<std::chrono::high_resolution_clock> pause_t1;
			std::chrono::time_point<std::chrono::high_resolution_clock> pause_t2;
			bool m_running;
			bool m_paused;
			double m_pauseOffset;
		};

		class NET_API DebugFunctionTime : public Timer
		{
			public:
			DebugFunctionTime(const std::string& funcName);
			~DebugFunctionTime();

			private:
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
#define CONSOLE_FUNCTION(x) ;
#define DEBUG_FUNCTION_TIME_INTERVAL ;
#endif

#define PRINT_ERROR(x) \
	STANDARD_CONSOLE_OUTPUT("Error: " <<__PRETTY_FUNCTION__<<" : "<<x<<"\n")


#define __VERIFY_RANGE_COMP1(min,var,max) if(min>var || var>max){ PRINT_ERROR(std::string(#var)<<" out of range: "<<min<<" > "<<#var<<" = "<<var<<" > "<<max)
	//#define VERIFY_RANGE(min,var,max) __VERIFY_RANGE_COMP1(min,var,max)}
#define VERIFY_RANGE(min,var,max,ret)__VERIFY_RANGE_COMP1(min,var,max) ret;}

#define __VERIFY_BOOL_COMP1(val,comp,message) if(val != comp){PRINT_ERROR(message)

	// val must equal comp to not throw an error
#define VERIFY_BOOL(val,comp,message,ret) __VERIFY_BOOL_COMP1(val,comp,message) ret;}

#define __VERIFY_VALID_PTR_COMP1(ptr, message) if(!ptr){PRINT_ERROR(#ptr<<" == nullptr "<<message)
//#define VERIFY_VALID_PTR(ptr, message) __VERIFY_VALID_PTR_COMP1(ptr,message)}
//#define VERIFY_VALID_PTR(ptr, message, ret) __VERIFY_VALID_PTR_COMP1(ptr,message) ret;}

#define PTR_CHECK_NULLPTR(ptr,ret) \
    if(ptr == nullptr) \
    {   \
        PRINT_ERROR(std::string(#ptr) + " is a nullptr") \
        ret; \
    }
};