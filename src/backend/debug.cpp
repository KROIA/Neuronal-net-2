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

		void Timer::setPauseTime(const std::chrono::nanoseconds& offset)
		{
			m_pauseOffset = offset;
		}
		void Timer::addPauseTime(const std::chrono::nanoseconds& delta)
		{
			m_pauseOffset += delta;
		}
		const std::chrono::nanoseconds& Timer::getPauseTime() const
		{
			return m_pauseOffset;
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

			STANDARD_CONSOLE_OUTPUT(m_stackSpace)
				STANDARD_CONSOLE_OUTPUT(m_functionName << " begin\n")
				t1 = getCurrentTimePoint();

		}
		DebugFunctionTime::~DebugFunctionTime()
		{
			auto t2 = getCurrentTimePoint();
			STANDARD_CONSOLE_OUTPUT(m_stackSpace)
				STANDARD_CONSOLE_OUTPUT(m_functionName << " end time: " << timeToString(getMillis(t2 - t1)) << "\n")
				if (__DBG_stackDepth != 0)
					--__DBG_stackDepth;
		}




		void StackElement::updatePauseTime()
		{
			std::chrono::nanoseconds pause(0);
			for (size_t i = 0; i < m_childs.size(); ++i)
			{
				m_childs[i]->updatePauseTime();
				pause += m_childs[i]->m_pause;
			}
			m_pause += pause;
			m_time -= Timer::getMicros(pause);
		}
		void StackElement::printResult(double timeSum)
		{
			double effectiveTime = m_time;
			for (size_t i = 0; i < m_childs.size(); ++i)
			{
				effectiveTime -= m_childs[i]->m_time;
			}
			std::string space(m_stackIndex, ' ');
			char indxBuf[10];
			char timeBuf[20];
			char timeEffBuf[20];
			char timePercentBuf[20];
			sprintf_s(indxBuf, "%5i", m_stackIndex);
			sprintf_s(timeBuf, "%10.3lf", m_time);
			sprintf_s(timeEffBuf, "%10.3lf", effectiveTime);
			sprintf_s(timePercentBuf, "%10.3lf", effectiveTime * 100.f / timeSum);
			STANDARD_CONSOLE_OUTPUT("StackIdx = " << indxBuf << " AbsTime = " << timeBuf << " us ")
			STANDARD_CONSOLE_OUTPUT("EffectiveTime = " << timeEffBuf << " us " << timePercentBuf << "% "<< space << m_context << "\n")
			for (size_t i = 0; i < m_childs.size(); ++i)
			{
				m_childs[i]->printResult(timeSum);
			}
		}


		size_t DebugFuncStackTimeTrace::m_standardStackSize = 50;
		size_t DebugFuncStackTimeTrace::m_standardChannelSize = 50;
		//std::vector<std::vector<DebugFuncStackTimeTrace*> > DebugFuncStackTimeTrace::m_stack(m_standardChanalSize);
		std::vector<std::vector<double> > DebugFuncStackTimeTrace::m_timeStack(m_standardChannelSize);
		std::vector<int > DebugFuncStackTimeTrace::m_stackDepth(m_standardChannelSize, 0);
		std::vector< std::vector<StackElement* > > DebugFuncStackTimeTrace::m_stack(m_standardChannelSize);
		DebugFuncStackTimeTrace::DebugFuncStackTimeTrace(const std::string& context, size_t channel)
		{
			//m_thisStack = nullptr;
			m_timer.start();
			m_timer.pause();
			m_channel = 0;
			m_timeIndex = 0;
			//m_thisTimeStack = nullptr;
			m_thisStack		= nullptr;
			if (channel >= m_standardChannelSize)
			{
				STANDARD_CONSOLE_OUTPUT(__PRETTY_FUNCTION__ << " chanal out of range. Max is: " << m_standardChannelSize)
				return;
			}
			
			m_channel = channel;
			//m_thisStack		= &m_stack[m_channel];
			//m_thisTimeStack = &m_timeStack[m_channel];
			m_thisStack     = &m_stack[m_channel];
			//if (m_thisStack->capacity() < m_standardStackSize)
			//	m_thisStack->reserve(m_standardStackSize);
			//if (m_thisTimeStack->capacity() < m_standardStackSize)
			//	m_thisTimeStack->reserve(m_standardStackSize);
			if (m_thisStack->capacity() < m_standardStackSize)
				m_thisStack->reserve(m_standardStackSize);
			//m_timeIndex = m_thisTimeStack->size();
			//m_thisStack->push_back(this);
			//m_thisTimeStack->push_back(0);

			if (m_thisStack->size() == 0)
				m_thisStackelement = new StackElement(context,0);
			else
			{
				m_thisStackelement = (*m_thisStack)[m_thisStack->size() - 1]->addChild(context);
			}
			m_thisStack->push_back(m_thisStackelement);
			m_timer.unpause();
			
		}
		DebugFuncStackTimeTrace::~DebugFuncStackTimeTrace()
		{
			//if (!m_thisTimeStack)
			//	return;
			m_timer.stop();
			m_thisStackelement->setTime(m_timer);
			
			//(*m_thisTimeStack)[m_timeIndex] = m_timer.getMicros();
			if (m_thisStack->size() == 1)
			{
				printResults();
				//m_thisStack->clear();
				//m_thisTimeStack->clear();
				delete m_thisStackelement;
				STANDARD_CONSOLE_OUTPUT("pause")
				getchar();
			}
			m_thisStack->pop_back();
		}

		void DebugFuncStackTimeTrace::printResults()
		{
			STANDARD_CONSOLE_OUTPUT(__PRETTY_FUNCTION__ << " Channel: " << m_channel << "\n")
			m_thisStackelement->updatePauseTime();
			m_thisStackelement->printResult(m_thisStackelement->getTime());

			//STANDARD_CONSOLE_OUTPUT("Individual times:\n")
			/*std::vector<double> individualTime(m_thisTimeStack->size(), 0);
			double timeSume = (*m_thisTimeStack)[0];
			for (size_t i = m_thisTimeStack->size(); i > 0; --i)
			{
				if (i == m_thisTimeStack->size())
				{
					individualTime[i - 1] = (*m_thisTimeStack)[i - 1];
				}
				else
				{
					individualTime[i - 1] = (*m_thisTimeStack)[i - 1] - individualTime[i];
				}
			}
			for (size_t i = 0; i < m_thisTimeStack->size(); ++i)
			{
				float gesTime = (*m_thisTimeStack)[i];
				float effectiveTime = individualTime[i];
				std::string space(i, ' ');
				char buff[10];
				sprintf_s(buff, "%5i", i);
				STANDARD_CONSOLE_OUTPUT("StackIdx = " << buff << space << " AbsTime = " << gesTime << " us ")
				STANDARD_CONSOLE_OUTPUT(" EffectiveTime = " << effectiveTime << " us " << effectiveTime *100.f / timeSume << "%\n")

			}*/
		}

	}

};