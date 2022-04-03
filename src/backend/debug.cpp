#include "backend/debug.h"


namespace NeuronalNet
{
	namespace Debug
	{
#ifdef UNIT_TEST
		std::vector<std::string> Debug::_unitTest_consoleBuffer;
#endif

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
				if (m || h)
					timeStr += " " + std::to_string(m) + " min";
				if (m || h || s)
					timeStr += " " + std::to_string(s) + " s";
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

		Timer::Timer(bool autoStart)
			: m_running(false)
			, m_paused(false)
			, m_pauseOffset(0)
		{
			if (autoStart)
				start();
		}
		Timer::~Timer()
		{

		}

		void Timer::start()
		{
			m_pauseOffset = std::chrono::nanoseconds(0);
			m_running = true;
			m_paused = false;
			t1 = std::chrono::high_resolution_clock::now();
		}
		void Timer::stop()
		{
			t2 = std::chrono::high_resolution_clock::now();
			if (m_paused)
			{
				pause_t2 = t2;
				m_pauseOffset += pause_t2 - pause_t1;
			}
			m_running = false;
			m_paused = false;
		}
		void Timer::pause()
		{
			if (!m_paused && m_running)
			{
				pause_t1 = std::chrono::high_resolution_clock::now();
				t2 = pause_t1;
				m_paused = true;
			}
		}
		void Timer::unpause()
		{
			if (m_paused && m_running)
			{
				pause_t2 = std::chrono::high_resolution_clock::now();
				m_pauseOffset += (pause_t2 - pause_t1);
				m_paused = false;
			}
		}
		double Timer::getMillis() const
		{
			if (!m_paused && m_running)
			{
				std::chrono::time_point<std::chrono::high_resolution_clock> current = std::chrono::high_resolution_clock::now();
				return getMillis(current - t1 - m_pauseOffset);
			}
			return getMillis(t2 - t1 - m_pauseOffset);
		}
		double Timer::getMicros() const
		{
			if (!m_paused && m_running)
			{
				std::chrono::time_point<std::chrono::high_resolution_clock> current = std::chrono::high_resolution_clock::now();
				return getMicros(current - t1 - m_pauseOffset);
			}
			return getMicros(t2 - t1 - m_pauseOffset);
		}
		double Timer::getNanos() const
		{
			if (!m_paused && m_running)
			{
				std::chrono::time_point<std::chrono::high_resolution_clock> current = std::chrono::high_resolution_clock::now();
				return getNanos(current - t1 - m_pauseOffset);
			}
			return getNanos(t2 - t1 - m_pauseOffset);
		}

		void Timer::reset()
		{
			m_running = false;
			m_paused  = false;
			t1 = t2;
			pause_t2 = pause_t1;
			m_pauseOffset = std::chrono::nanoseconds(0);
		}
		bool Timer::isRunning() const
		{
			return m_running;
		}
		bool Timer::isPaused() const
		{
			return m_paused;
		}

		inline std::chrono::time_point<std::chrono::high_resolution_clock> Timer::getCurrentTimePoint()
		{
			return std::chrono::high_resolution_clock::now();
		}
		template <typename T>
		inline double Timer::getMillis(T t)
		{
			return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t).count() / 1000000;
		}
		template <typename T>
		inline double Timer::getMicros(T t)
		{
			return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t).count() / 1000;
		}
		template <typename T>
		inline double Timer::getNanos(T t)
		{
			return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t).count();
		}


		size_t __DBG_stackDepth = 0;

		DebugFunctionTime::DebugFunctionTime(const std::string& funcName)
		{


			m_stackSpace.resize(__DBG_stackDepth * 2, ' ');
			m_functionName = funcName;
			++__DBG_stackDepth;

			CONSOLE_RAW(m_stackSpace)
				CONSOLE_RAW(m_functionName << " begin\n")
				t1 = getCurrentTimePoint();

		}
		DebugFunctionTime::~DebugFunctionTime()
		{
			auto t2 = getCurrentTimePoint();
			CONSOLE_RAW(m_stackSpace)
				CONSOLE_RAW(m_functionName << " end time: " << timeToString(getMillis(t2 - t1)) << "\n")
				if (__DBG_stackDepth != 0)
					--__DBG_stackDepth;
		}

	}

};