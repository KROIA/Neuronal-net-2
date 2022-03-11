#pragma once

#include "graphicsUtilities.h"
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

			
			void pos(const sf::Vector2f& pos);
			const sf::Vector2f& pos() const;
			void size(float size);
			float size() const;

			


			// Interface implementation
			void draw(sf::RenderWindow* window,
					  const sf::Vector2f& offset = sf::Vector2f(0, 0));

			virtual void update(float netinput, float output);

			static const float standardSize;
			protected:

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







