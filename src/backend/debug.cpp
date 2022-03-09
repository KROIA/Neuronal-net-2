#include "backend/debug.h"



namespace Debug
{
#ifdef UNIT_TEST
	std::vector<std::string> Debug::_unitTest_consoleBuffer;
#endif
	// helper functions for cleaner time measuring code
	std::chrono::time_point<std::chrono::high_resolution_clock> now() {
		return std::chrono::high_resolution_clock::now();
	}

	template <typename T>
	double milliseconds(T t) {
		return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t).count() / 1000000;
	}

	std::string timeToString(double timeMs)
	{
		if (timeMs <= 0)
			return "0 ms";

		std::string timeStr = "";

		if (timeMs < 1)
		{
			timeMs *= 1000;
			int us = timeMs;
			timeMs -= (double)us;
			timeMs *= 1000;
			int ns = timeMs;
			timeMs -= (double)ns;

			timeStr = std::to_string(us) + " us";
			if (ns)
				timeStr += " " + std::to_string(ns) + " ns";
		}
		else
		{
			size_t ms = (size_t)timeMs % 1000;
			//timeMs = (timeMs-ms) * 1000.l;
			timeMs /= 1000;
			size_t rawS = (size_t)timeMs;

			size_t s = rawS % 60;
			size_t m = (rawS / 60) % 60;
			size_t h = rawS / 3600;

			if (h)
				timeStr = std::to_string(h) + " h";
			if(m || h)
				timeStr += " "+std::to_string(m) + " min";
			if (m || h || s)
				timeStr += " "+std::to_string(s) + " s";
			timeStr += " " + std::to_string(ms) + " ms";
			
		}
		return timeStr;
	}
	std::string bytesToString(size_t byteCount)
	{
		if (byteCount == 0)
			return "0 byte";
		int exp = log10((double)byteCount);
		int kStep = exp / 3;
		if (kStep > 3)
			kStep = 3;
		char number[20];
		std::string valueStr;
		if (kStep > 0)
		{
			sprintf(number, "%4.3f", byteCount / pow(1000, kStep));
			valueStr = number;
		}
		else
			valueStr = std::to_string(byteCount);

		switch (kStep)
		{
			case 0: return valueStr + " byte";
			case 1: return valueStr + " KB";
			case 2: return valueStr + " MB";
			case 3:
			default:
				return valueStr + " GB";
		}
		//cout << "byteCount: " << byteCount << " exp: " << exp << " kStep: " << kStep << " " << valueStr << "\n";
		return "";
	}


	size_t __DBG_stackDepth = 0;

	DebugFunctionTime::DebugFunctionTime(const std::string& funcName)
	{
		
		
		m_stackSpace.resize(__DBG_stackDepth*2, ' ');
		m_functionName = funcName;
		++__DBG_stackDepth;
		
		CONSOLE_RAW(m_stackSpace)
		CONSOLE_RAW(m_functionName<<" begin\n")
		t1 = now();
		
	}
	DebugFunctionTime::~DebugFunctionTime()
	{
		auto t2 = now();
		CONSOLE_RAW(m_stackSpace)
		CONSOLE_RAW(m_functionName<<" end time: " << timeToString(milliseconds(t2 - t1))<<"\n")
		if (__DBG_stackDepth != 0)
			--__DBG_stackDepth;
	}

}