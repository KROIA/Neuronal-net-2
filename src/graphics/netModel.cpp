#include "graphics/netModel.h"

namespace NeuronalNet
{
	namespace Graphics
	{
		NetModel::NetModel(Net* net)
		{
			m_net = net;
			rebuild();
		}
		NetModel::~NetModel()
		{
			clear();
		}

		void NetModel::rebuild()
		{
			clear();
			if (!m_net)
				return;
			if (!m_net->isBuilt())
			{
				CONSOLE_FUNCTION("Net is not built")
				return;
			}

			sf::Vector2f position(100, 100);
			sf::Vector2f deltaPos(100,100);
			size_t maxYSize = m_net->getInputCount();
			if (maxYSize < m_net->getHiddenYCount())
				maxYSize = m_net->getHiddenYCount();
			if (maxYSize < m_net->getOutputCount())
				maxYSize = m_net->getOutputCount();

			sf::Vector2f inputOffset( position.x, position.y+ (deltaPos.y * (maxYSize - m_net->getInputCount()))/2);
			sf::Vector2f hiddenOffset(position.x + deltaPos.x,position.y+ (deltaPos.y * (maxYSize - m_net->getHiddenYCount()))/2);
			sf::Vector2f outputOffset(position.x + deltaPos.x * (m_net->getHiddenXCount()+1),position.y+ (deltaPos.y * (maxYSize - m_net->getOutputCount()))/2);
			float size = 20;
			for (size_t y = 0; y < m_net->getInputCount(); ++y)
			{
				sf::Vector2f pos(inputOffset.x, inputOffset.y + deltaPos.y * y);
				NeuronIndex index;
				index.type = NeuronType::input;
				index.x = 0;
				index.y = y;
				Neuron* neuron = new Neuron();
				neuron->index(index);
				neuron->size(size);
				neuron->pos(pos);
				m_neuronList.push_back(neuron);
			}

			for (size_t x = 0; x < m_net->getHiddenXCount(); ++x)
			{
				for (size_t y = 0; y < m_net->getHiddenYCount(); ++y)
				{
					sf::Vector2f pos(hiddenOffset.x + deltaPos.x * x, hiddenOffset.y + deltaPos.y * y);
					NeuronIndex index;
					index.type = NeuronType::hidden;
					index.x = x;
					index.y = y;
					Neuron* neuron = new Neuron();
					neuron->index(index);
					neuron->size(size);
					neuron->pos(pos);
					m_neuronList.push_back(neuron);
				}
			}

			for (size_t y = 0; y < m_net->getOutputCount(); ++y)
			{
				sf::Vector2f pos(outputOffset.x, outputOffset.y + deltaPos.y * y);
				NeuronIndex index;
				index.type = NeuronType::output;
				index.x = 0;
				index.y = y;
				Neuron* neuron = new Neuron();
				neuron->index(index);
				neuron->size(size);
				neuron->pos(pos);
				m_neuronList.push_back(neuron);
			}
			for (size_t i = 0; i < m_neuronList.size(); ++i)
				m_net->addGraphics(m_neuronList[i]);
		}

		void NetModel::draw(sf::RenderWindow* window,
							const sf::Vector2f &offset)
		{
			for (size_t i = 0; i < m_neuronList.size(); ++i)
				m_neuronList[i]->draw(window, offset);
		}

		void NetModel::clear()
		{
			for (size_t i = 0; i < m_neuronList.size(); ++i)
			{
				m_net->removeGraphics(m_neuronList[i]);
				delete m_neuronList[i];
			}
			m_neuronList.clear();
		}
	};
};