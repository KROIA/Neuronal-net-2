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

			m_window = new sf::RenderWindow(sf::VideoMode(m_windowSize.x, m_windowSize.y),
											m_windowTitle);
		}
		Display::~Display()
		{
			delete m_window;
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
		void Display::draw()
		{
			m_window->clear();
			for (size_t i = 0; i < m_drawableObjList.size(); ++i)
				m_drawableObjList[i]->draw(m_window);
			m_window->display();
		}
	};
};