#include "graphics/display.h"


namespace NeuronalNet
{
	namespace Graphics
	{

		Display::Display()
		{
			init(sf::Vector2u(800, 800), "Window");
		}
		Display::Display(sf::Vector2u size, const string& title)
		{
			init(size, title);
		}
		void Display::init(sf::Vector2u size, const string& title)
		{
			m_windowSize = size;
			m_windowTitle = title;
			m_exit = false;
			m_targetFrameTimeMs = 1000.f/60.f;
			m_frameIntervalTime = nullptr;

			m_window = new sf::RenderWindow(sf::VideoMode(m_windowSize.x, m_windowSize.y),
											m_windowTitle);
		}
		Display::~Display()
		{
			delete m_window;
			if (m_frameIntervalTime)
				delete m_frameIntervalTime;
		}
		void Display::addDrawable(Drawable* obj)
		{
			if (obj == nullptr)
				return;
			for (size_t i = 0; i < m_drawableObjList.size(); ++i)
				if (m_drawableObjList[i] == obj)
					return;
			m_drawableObjList.push_back(obj);
		}

		void Display::frameRateTarget(float fps)
		{
			if(fps > 0)
				m_targetFrameTimeMs = 1000 / fps;
		}
		bool Display::isOpen()
		{
			return m_window->isOpen() && !m_exit;
		}
		void Display::loop()
		{
			while (isOpen())
			{
				processEvents();
				draw();
			}
		}
		void Display::processEvents()
		{
			sf::Event event;
			while (m_window->pollEvent(event))
			{
				// Close window: exit
				if (event.type == sf::Event::Closed)
					m_window->close();
			}
		}
		bool Display::needsFrameUpdate()
		{
			// Check if frame update is needed
			if (!m_frameIntervalTime)
				return true;
			std::chrono::time_point<std::chrono::high_resolution_clock> t2 =
				std::chrono::high_resolution_clock::now();
			double ms = (double)std::chrono::duration_cast<std::chrono::nanoseconds>
				(t2 - *m_frameIntervalTime).count() / 1000000.f;
			if (ms < m_targetFrameTimeMs)
				return false;
			*m_frameIntervalTime = t2;
			return true;
		}
		void Display::draw()
		{
			//if (!needsFrameUpdate())
			//	return;
			if (!m_frameIntervalTime)
			{
				m_frameIntervalTime = new std::chrono::time_point<std::chrono::high_resolution_clock>();
				*m_frameIntervalTime = std::chrono::high_resolution_clock::now();
			}
			// Make frame update
			m_window->clear();
			for (size_t i = 0; i < m_drawableObjList.size(); ++i)
				m_drawableObjList[i]->draw(m_window);
			m_window->display();

			

			
		
			
		}
	};
};