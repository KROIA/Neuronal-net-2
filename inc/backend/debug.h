#pragma once
#define _CRTDBG_MAP_ALLOC
#include <cstdlib>
#include <crtdbg.h>

#include <iostream>
#include <stdio.h>
#include <string>
#include <chrono>

#include "config.h"
#include <vector>

#ifdef UNIT_TEST

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
			double getMicros() const;
			double getNanos() const;
			void reset();
			bool isRunning() const;
			bool isPaused() const;

			void setPauseTime(const std::chrono::nanoseconds& offset);
			void addPauseTime(const std::chrono::nanoseconds& delta);
			const std::chrono::nanoseconds& getPauseTime() const;

			static inline std::chrono::time_point<std::chrono::high_resolution_clock> getCurrentTimePoint();
			template <typename T>
			static inline double getMillis(T t);
			template <typename T>
			static inline double getMicros(T t);
			template <typename T>
			static inline double getNanos(T t);

			protected:
			std::chrono::time_point<std::chrono::high_resolution_clock> t1;
			std::chrono::time_point<std::chrono::high_resolution_clock> t2;

			std::chrono::time_point<std::chrono::high_resolution_clock> pause_t1;
			std::chrono::time_point<std::chrono::high_resolution_clock> pause_t2;
			std::chrono::nanoseconds m_pauseOffset;
			bool m_running;
			bool m_paused;
			//double m_pauseOffset;
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
		class NET_API StackElement
		{
			public:
			StackElement(const std::string& context, int stackIndex) {
				m_stackIndex = stackIndex;
				m_time = 0;
				m_childs.reserve(20);
				m_context = context;
			}
			~StackElement() {
				for (StackElement* el : m_childs)
					delete el;
				m_childs.clear();
			}

			inline StackElement* addChild(const std::string& context){
				StackElement* el = new StackElement(context,m_stackIndex + 1);
				m_childs.push_back(el);
				return el;
			}

			inline void setTime(const Timer &t) {
				m_time = t.getMicros();
				m_pause = t.getPauseTime();
			}
			inline double getTime() const {
				return m_time;
			}
			inline int getStackIndex() const {
				return m_stackIndex;
			}
			
			void updatePauseTime();
			void printResult(double timeSum);

			private:

			double m_time;
			int m_stackIndex;
			std::string m_context;
			std::chrono::nanoseconds m_pause;
			std::vector<StackElement*> m_childs;

		};
		class NET_API DebugFuncStackTimeTrace
		{
			public:
			DebugFuncStackTimeTrace(const std::string &context, size_t channel = 0);
			~DebugFuncStackTimeTrace();

			void printResults();

			protected:
			Timer m_timer;
			size_t m_timeIndex;
			size_t m_channel;
			//std::vector<DebugFuncStackTimeTrace*>* m_thisStack;
			//std::vector<double>* m_thisTimeStack;
			std::vector<StackElement*>* m_thisStack;
			StackElement* m_thisStackelement;



			static size_t m_standardStackSize;
			static size_t m_standardChannelSize;
			static std::vector<std::vector<double> > m_timeStack;
			static std::vector<int > m_stackDepth;
			static std::vector< std::vector<StackElement*> > m_stack;
			//static std::vector<std::vector<DebugFuncStackTimeTrace*> > m_stack;
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