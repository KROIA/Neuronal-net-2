#pragma once

#include "graphicsUtilities.h"
#include "drawable.h"
#include "backend/graphicsNeuronInterface.h"



namespace NeuronalNet
{
	namespace Graphics
	{
		using std::string;

		class NET_API NeuronPainter	:	public Drawable, public GraphicsNeuronInterface
		{
			public:
			NeuronPainter();
			NeuronPainter(const NeuronPainter& other);
			~NeuronPainter();

			const NeuronPainter& operator=(const NeuronPainter& other);

			static inline size_t getStandardVisualConfiguration();

			void setPos(const sf::Vector2f& pos);
			const sf::Vector2f& getPos() const;
			void setSize(float size);
			float getSize() const;

			//virtual void setOptimization(Optimization opt);
			


			// Interface implementation
			void draw(sf::RenderWindow* window,
					  const sf::Vector2f& offset = sf::Vector2f(0, 0));

			virtual void update(float netinput, float output,
								float minN, float maxN,
								float minO, float maxO);

			static const float standardSize;
			protected:

			sf::Color m_color;
			sf::Text  m_outputText;
			sf::Vector2f m_pos;
			float m_size;

			sf::CircleShape m_circleShape;

			static const string m_fontPath;
			static sf::Font m_font;

			// Data from Net
			float m_netinput;
			float m_output;

			// Performance settings
			//bool m_useTextLabel; 
			
		};
	};
};







