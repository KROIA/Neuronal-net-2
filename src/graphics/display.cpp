#include "graphics/display.h"


namespace NeuronalNet
{
	namespace Graphics
	{

		Display::Display()
			: m_window(nullptr)
			, m_frameIntervalTime(nullptr)
		{
			init(sf::Vector2u(800, 800), "Window");
		}
		Display::Display(sf::Vector2u size, const string& title)
			: m_window(nullptr)
			, m_frameIntervalTime(nullptr)
		{
			init(size, title);
		}
		void Display::init(sf::Vector2u size, const string& title)
		{
			m_windowSize = size;
			m_windowTitle = title;
			m_exit = false;
			m_targetFrameTimeMs = 1000.f / 60.f;

			if(!m_frameIntervalTime)
			m_frameIntervalTime = DBG_NEW std::chrono::time_point<std::chrono::high_resolution_clock>();
			*m_frameIntervalTime = std::chrono::high_resolution_clock::now();

			setSizeFixed(false);
			if(!m_window)
			m_window = DBG_NEW sf::RenderWindow(sf::VideoMode(m_windowSize.x, m_windowSize.y),
											m_windowTitle);
		}
		Display::~Display()
		{
			delete m_window;
			if (m_frameIntervalTime)
				delete m_frameIntervalTime;
		}

		void Display::setSizeFixed(bool enable)
		{
			m_fixedWindowSize = enable;
		}

		void Display::addSubscriber(DisplayInterface* subscriber)
		{
			PTR_CHECK_NULLPTR(subscriber, return)
			VECTOR_INSERT_ONCE(m_subscriber, subscriber)
		}
		void Display::removeSubsctiber(DisplayInterface* subscriber)
		{
			PTR_CHECK_NULLPTR(subscriber, return)
			VECTOR_REMOVE_ELEM(m_subscriber, subscriber)
		}
		void Display::clearSubscriber()
		{
			m_subscriber.clear();
		}

		void Display::addDrawable(DrawableInterface* drawable, size_t layer)
		{
			PTR_CHECK_NULLPTR(drawable, return)
			while (m_renderLayer.size() <= layer)
			{
				m_renderLayer.push_back(RenderLayer{ true,
													 vector<DrawableInterface*>() });
			}
			VECTOR_INSERT_ONCE(m_renderLayer[layer].painter, drawable)
		}
		void Display::removeDrawable(DrawableInterface* drawable, size_t layer)
		{
			if (layer >= m_renderLayer.size())
			{
				PRINT_ERROR("layer is out of range\nlayer = " + std::to_string(layer) + "\nmax is = " + std::to_string(m_renderLayer.size()))
					return;
			}
			PTR_CHECK_NULLPTR(drawable, return)
				VECTOR_REMOVE_ELEM(m_renderLayer[layer].painter, drawable)
		}
		void Display::clearDrawable()
		{
			m_renderLayer.clear();
		}
		void Display::clearDrawable(size_t layer)
		{
			if (layer >= m_renderLayer.size())
			{
				PRINT_ERROR("layer is out of range\nlayer = " + std::to_string(layer) + "\nmax is = " + std::to_string(m_renderLayer.size()))
					return;
			}
			m_renderLayer[layer].painter.clear();
		}

		void Display::exitLoop()
		{
			m_exit = true;
		}
		void Display::setLayerVisibility(size_t layer, bool visible)
		{
			if (layer >= m_renderLayer.size())
			{
				PRINT_ERROR("layer is out of range\nlayer = " + std::to_string(layer) + "\nmax is = " + std::to_string(m_renderLayer.size()))
					return;
			}
			m_renderLayer[layer].isVisible = visible;
		}
		void Display::toggleLayerVisibility(size_t layer)
		{
			if (layer >= m_renderLayer.size())
			{
				PRINT_ERROR("layer is out of range\nlayer = " + std::to_string(layer) + "\nmax is = " + std::to_string(m_renderLayer.size()))
					return;
			}
			m_renderLayer[layer].isVisible = !m_renderLayer[layer].isVisible;
		}
		bool Display::getLayerVisibility(size_t layer) const
		{
			if (layer >= m_renderLayer.size())
			{
				PRINT_ERROR("layer is out of range\nlayer = " + std::to_string(layer) + "\nmax is = " + std::to_string(m_renderLayer.size()))
					return false;
			}
			return m_renderLayer[layer].isVisible;
		}

		sf::Vector2i Display::getRelativeMousePos() const
		{
			const sf::Vector2i ribbon(0, 25);
			return sf::Mouse::getPosition() - m_window->getPosition() - ribbon;
		}
		void Display::setSize(const sf::Vector2u& size)
		{
			m_windowSize = size;
			m_window->setSize(m_windowSize);
		}
		sf::Vector2u Display::getSize() const
		{
			return m_windowSize;
		}
		void Display::setTitle(const string& title)
		{
			m_windowTitle = title;
			m_window->setTitle(m_windowTitle);
		}
		const string& Display::getTitle() const
		{
			return m_windowTitle;
		}

		void Display::frameRateTarget(float fps)
		{
			if (fps > 0)
				m_targetFrameTimeMs = 1000 / fps;
		}
		bool Display::isOpen()
		{
			return m_window->isOpen();
		}
		void Display::loop()
		{
			m_exit = false;
			while (isOpen() && !m_exit)
			{
				onLoop();
				processEvents();
				draw();
			}
		}
		void Display::processEvents()
		{
			onPreEventUpdate();
			sf::Event event;
			while (m_window->pollEvent(event))
			{
				// Close window: exit
				switch (event.type)
				{
					case sf::Event::Closed:
						onDisplyCloseEvent();
						break;
					case sf::Event::KeyPressed:
						onKeyEvent(event);
						onKeyPressEvent(event);
						break;
					case sf::Event::KeyReleased:
						onKeyEvent(event);
						onKeyReleaseEvent(event);
						break;
					case sf::Event::MouseButtonPressed:
					case sf::Event::MouseButtonReleased:
					case sf::Event::MouseMoved:
					case sf::Event::MouseWheelScrolled:
					case sf::Event::MouseWheelMoved:
						onMouseEvent(event);
						break;
					case sf::Event::Resized:
						if (m_fixedWindowSize)
							m_window->setSize(m_windowSize);
						break;
				}
				onEvent(event);
			}
			onPostEventUpdate();
		}
		bool Display::needsFrameUpdate()
		{
			// Check if frame update is needed
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
			onPreFrameUpdate();
			// Make frame update
			m_window->clear();
			for (size_t layer = 0; layer < m_renderLayer.size(); ++layer)
				if (m_renderLayer[layer].isVisible)
					for (size_t i = 0; i < m_renderLayer[layer].painter.size(); ++i)
						m_renderLayer[layer].painter[i]->displayCallDraw(m_window);
			m_window->display();
			onPostFrameUpdate();
		}

		void Display::onPreEventUpdate()
		{
			for (DisplayInterface* sub : m_subscriber)
				sub->onPreEventUpdate();
		}
		void Display::onEvent(const sf::Event& event)
		{
			for (DisplayInterface* sub : m_subscriber)
				sub->onEvent(event);
		}
		void Display::onKeyEvent(const sf::Event& event)
		{
			for (DisplayInterface* sub : m_subscriber)
				sub->onKeyEvent(event);
		}
		void Display::onKeyPressEvent(const sf::Event& event)
		{
			for (DisplayInterface* sub : m_subscriber)
				sub->onKeyPressEvent(event);
		}
		void Display::onKeyReleaseEvent(const sf::Event& event)
		{
			for (DisplayInterface* sub : m_subscriber)
				sub->onKeyReleaseEvent(event);
		}
		void Display::onMouseEvent(const sf::Event& event)
		{
			for (DisplayInterface* sub : m_subscriber)
				sub->onMouseEvent(event);
		}
		void Display::onDisplyCloseEvent()
		{
			for (DisplayInterface* sub : m_subscriber)
				sub->onDisplyCloseEvent();
			m_window->close();
		}
		void Display::onPostEventUpdate()
		{
			for (DisplayInterface* sub : m_subscriber)
				sub->onPostEventUpdate();
		}
		void Display::onPreFrameUpdate()
		{
			for (DisplayInterface* sub : m_subscriber)
				sub->onPreFrameUpdate();
		}
		void Display::onPostFrameUpdate()
		{
			for (DisplayInterface* sub : m_subscriber)
				sub->onPostFrameUpdate();
		}
		void Display::onLoop()
		{
			for (DisplayInterface* sub : m_subscriber)
				sub->onLoop();
		}


	};
};