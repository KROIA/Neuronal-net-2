#pragma once

#include <iostream>
#include <SFML/Graphics.hpp>
#include <string>

#include "backend/debug.h"
#include "backend/config.h"
#include "drawable.h"
#include "backend/graphicsNeuronInterface.h"



namespace NeuronalNet
{
	namespace Graphics
	{
		using std::string;

		class NET_API Neuron	:	public Drawable, public GraphicsNeuronInterface
		{
			public:
			Neuron();
			Neuron(const Neuron& other);
			~Neuron();

			const Neuron& operator=(const Neuron& other);

			void index(const NeuronIndex& index);
			void pos(const sf::Vector2f& pos);
			const sf::Vector2f& pos() const;
			void size(float size);
			float size() const;

			void draw(sf::RenderWindow* window,
					  const sf::Vector2f &offset = sf::Vector2f(0, 0));


			// Interface implementation
			virtual void update(float netinput, float output);
			const virtual NeuronIndex& index() const;

			protected:
			void setColor(float output);

			NeuronIndex m_index;
			sf::Color m_color;
			sf::Text  m_outputText;
			sf::Vector2f m_pos;
			float m_size;

			static const string m_fontPath;
			sf::Font m_font;

			// Data from Net
			float m_netinput;
			float m_output;
			
		};
	};
};







