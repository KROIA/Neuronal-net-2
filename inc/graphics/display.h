#pragma once

#include <SFML/Graphics.hpp>
#include <string>
#include <vector>
#include <chrono>
#include "backend/config.h"
#include "backend/debug.h"
#include "backend/utilities.h"

#include "graphics/drawableInterface.h"
#include "graphics/displayInterface.h"


namespace NeuronalNet
{
	namespace Graphics
	{
		using std::string;
		using std::vector;

		struct RenderLayer
		{
			bool isVisible;
			vector<DrawableInterface*> painter;
		};

		
		class NET_API Display
		{
			public:
			Display();
			Display(sf::Vector2u size, const string& title);
			~Display();

			void setSizeFixed(bool enable);

			void addSubscriber(DisplayInterface* subscriber);
			void removeSubsctiber(DisplayInterface* subscriber);
			void clearSubscriber();

			void addDrawable(DrawableInterface* drawable, size_t layer = 0);
			void removeDrawable(DrawableInterface* drawable, size_t layer = 0);
			void clearDrawable();
			void clearDrawable(size_t layer);

			void setLayerVisibility(size_t layer, bool visible);
			void toggleLayerVisibility(size_t layer);
			bool getLayerVisibility(size_t layer) const;

			sf::Vector2i getRelativeMousePos() const;

			void setSize(const sf::Vector2u& size);
			sf::Vector2u getSize() const;
			void setTitle(const string& title);
			const string& getTitle() const;

			void exitLoop();

			void frameRateTarget(float fps);
			bool isOpen();
			void loop();
			void processEvents();
			bool needsFrameUpdate();
			void draw();



			protected:
			virtual void init(sf::Vector2u size, const string& title);

			void onPreEventUpdate();
			void onEvent(const sf::Event& event);
			void onKeyEvent(const sf::Event& event);
			void onKeyPressEvent(const sf::Event& event);
			void onKeyReleaseEvent(const sf::Event& event);
			void onMouseEvent(const sf::Event& event);
			void onDisplyCloseEvent();
			void onPostEventUpdate();

			void onPreFrameUpdate();
			void onPostFrameUpdate();

			void onLoop();


			private:
			sf::RenderWindow* m_window;
			sf::Vector2u m_windowSize;
			string m_windowTitle;

			bool m_fixedWindowSize;
			bool m_exit;
			vector<RenderLayer> m_renderLayer;

			double m_targetFrameTimeMs;
			std::chrono::time_point<std::chrono::high_resolution_clock>* m_frameIntervalTime;


			vector<DisplayInterface*> m_subscriber;
		};
	};
};