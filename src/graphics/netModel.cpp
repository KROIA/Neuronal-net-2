#include "graphics/netModel.h"

namespace NeuronalNet
{
	namespace Graphics
	{
		NetModel::NetModel(Net* net)
		{
			m_net				= net;
			m_connectionWidth	= Connection::standardConnectionWidth;
			m_signalWidth		= Connection::standardSignalWidth;
			m_neuronSize		= Neuron::standardSize;
			m_neuronSpacing		= sf::Vector2f(50, 50);
			m_streamIndex = 0;
			rebuild();
		}
		NetModel::~NetModel()
		{
			clear();
		}

		void NetModel::streamIndex(size_t index)
		{
			m_streamIndex = index;
			if (m_streamIndex >= m_net->getStreamSize())
				m_streamIndex = m_net->getStreamSize() - 1;


		}
		size_t NetModel::streamIndex() const
		{
			return m_streamIndex;
		}

		void NetModel::pos(const sf::Vector2f& pos)
		{
			sf::Vector2f deltaPos = pos - m_pos;
			
			for (size_t i = 0; i < m_neuronList.size(); ++i)
				m_neuronList[i]->pos(m_neuronList[i]->pos() + deltaPos);
			m_pos = pos;
		}
		const sf::Vector2f& NetModel::pos() const
		{
			return m_pos;
		}
		void NetModel::neuronSpacing(const sf::Vector2f& sp)
		{
			m_neuronSpacing = sp;
			updateNeuronDimensions();
		}
		const sf::Vector2f& NetModel::neuronSpacing()
		{
			return m_neuronSpacing;
		}

		void NetModel::rebuild()
		{
			if (!m_net)
				return;
			if (!m_net->isBuilt())
			{
				PRINT_ERROR("Net is not built")
				return;
			}
			clear();

			for (size_t y = 0; y < m_net->getInputCount(); ++y)
			{
				NeuronIndex index;
				index.type = NeuronType::input;
				index.x = 0;
				index.y = y;
				Neuron* neuron = new Neuron();
				neuron->index(index);
				m_neuronList.push_back(neuron);
				m_inputNeurons.push_back(neuron);
			}

			for (size_t x = 0; x < m_net->getHiddenXCount(); ++x)
			{
				m_hiddenNeurons.push_back(vector<Neuron*>());
				for (size_t y = 0; y < m_net->getHiddenYCount(); ++y)
				{
					NeuronIndex index;
					index.type = NeuronType::hidden;
					index.x = x;
					index.y = y;
					Neuron* neuron = new Neuron();
					neuron->index(index);
					m_neuronList.push_back(neuron);
					m_hiddenNeurons[x].push_back(neuron);

					// Build connection
					if (x == 0)
					{
						for (size_t j = 0; j < m_net->getInputCount(); ++j)
						{
							ConnectionIndex conIndex;
							conIndex.neuron = index;
							conIndex.inputConnection = j;
							Connection* connection = new Connection(m_inputNeurons[j],
																	neuron);

							connection->index(conIndex);
							m_connectionList.push_back(connection);
						}
					}
					else
					{
						for (size_t j = 0; j < m_net->getHiddenYCount(); ++j)
						{
							ConnectionIndex conIndex;
							conIndex.neuron = index;
							conIndex.inputConnection = j;
							Connection* connection = new Connection(m_hiddenNeurons[x-1][j],
																	neuron);

							connection->index(conIndex);
							m_connectionList.push_back(connection);
						}
					}

				}
			}

			for (size_t y = 0; y < m_net->getOutputCount(); ++y)
			{
				NeuronIndex index;
				index.type = NeuronType::output;
				index.x = 0;
				index.y = y;
				Neuron* neuron = new Neuron();
				neuron->index(index);
				m_neuronList.push_back(neuron);
				m_outputNeurons.push_back(neuron);

				// Build connection
				if (m_net->getHiddenXCount() == 0)
				{
					for (size_t j = 0; j < m_net->getInputCount(); ++j)
					{
						ConnectionIndex conIndex;
						conIndex.neuron = index;
						conIndex.inputConnection = j;
						Connection* connection = new Connection(m_inputNeurons[j],
																neuron);

						connection->index(conIndex);
						m_connectionList.push_back(connection);
					}
				}
				else
				{
					for (size_t j = 0; j < m_net->getHiddenYCount(); ++j)
					{
						ConnectionIndex conIndex;
						conIndex.neuron = index;
						conIndex.inputConnection = j;
						Connection* connection = new Connection(m_hiddenNeurons[m_net->getHiddenXCount() - 1][j],
																neuron);

						connection->index(conIndex);
						m_connectionList.push_back(connection);
					}
				}
				
			}
			for (size_t i = 0; i < m_neuronList.size(); ++i)
				m_neuronInterface.push_back(m_neuronList[i]);
			for (size_t i = 0; i < m_connectionList.size(); ++i)
				m_connectionInterface.push_back(m_connectionList[i]);

			/*for (size_t i = 0; i < m_neuronList.size(); ++i)
				m_net->addGraphics(m_neuronList[i]);

			for (size_t i = 0; i < m_connectionList.size(); ++i)
				m_net->addGraphics(m_connectionList[i]);*/

			connectionWidth(m_connectionWidth);
			signalWidth(m_signalWidth);
			updateNeuronDimensions();
		}

		void NetModel::connectionWidth(float w)
		{
			m_connectionWidth = w;
			for (size_t i = 0; i < m_connectionList.size(); ++i)
				m_connectionList[i]->connectionWidth(m_connectionWidth);
		}
		void NetModel::signalWidth(float w)
		{
			m_signalWidth = w;
			for (size_t i = 0; i < m_connectionList.size(); ++i)
				m_connectionList[i]->signalWidth(m_signalWidth);
		}
		void NetModel::neuronSize(float size)
		{
			m_neuronSize = size;
			//internal_neuronSize(size);
			updateNeuronDimensions();
		}
		/*void NetModel::internal_neuronSize(float size)
		{
			for (size_t i = 0; i < m_neuronList.size(); ++i)
				m_neuronList[i]->size(m_neuronSize);
		}*/


		float NetModel::connectionWidth() const
		{
			return m_connectionWidth;
		}
		float NetModel::signalWidth() const
		{
			return m_signalWidth;
		}
		float NetModel::neuronSize() const
		{
			return m_neuronSize;
		}

		void NetModel::draw(sf::RenderWindow* window,
							const sf::Vector2f &offset)
		{
			m_net->graphics_update(m_neuronInterface, m_connectionInterface,m_streamIndex);
			for (size_t i = 0; i < m_connectionList.size(); ++i)
				m_connectionList[i]->draw(window, offset);
			for (size_t i = 0; i < m_neuronList.size(); ++i)
				m_neuronList[i]->draw(window, offset);
		}

		void NetModel::clear()
		{
			for (size_t i = 0; i < m_neuronList.size(); ++i)
			{
				//m_net->removeGraphics(m_neuronList[i]);
				delete m_neuronList[i];
			}
			m_neuronList.clear();

			for (size_t i = 0; i < m_neuronList.size(); ++i)
			{
				//m_net->removeGraphics(m_connectionList[i]);
				delete m_connectionList[i];
			}
			m_neuronInterface.clear();
			m_connectionInterface.clear();
			m_connectionList.clear();
			m_inputNeurons.clear();
			m_hiddenNeurons.clear();
			m_outputNeurons.clear();
		}
		void NetModel::updateNeuronDimensions()
		{
			
			size_t maxYSize = m_net->getInputCount();
			if (maxYSize < m_net->getHiddenYCount())
				maxYSize = m_net->getHiddenYCount();
			if (maxYSize < m_net->getOutputCount())
				maxYSize = m_net->getOutputCount();

			sf::Vector2f deltaPos(m_neuronSpacing.x + m_neuronSize*2, m_neuronSpacing.y + m_neuronSize*2);

			sf::Vector2f inputOffset(m_pos.x, m_pos.y + (deltaPos.y * (maxYSize - m_net->getInputCount())) / 2);
			sf::Vector2f hiddenOffset(m_pos.x + deltaPos.x, m_pos.y + (deltaPos.y * (maxYSize - m_net->getHiddenYCount())) / 2);
			sf::Vector2f outputOffset(m_pos.x + deltaPos.x * (m_net->getHiddenXCount() + 1), m_pos.y + (deltaPos.y * (maxYSize - m_net->getOutputCount())) / 2);

			for (size_t y = 0; y < m_inputNeurons.size(); ++y)
			{
				sf::Vector2f pos(inputOffset.x, inputOffset.y + deltaPos.y * y);
				m_inputNeurons[y]->size(m_neuronSize);
				m_inputNeurons[y]->pos(pos);
			}

			for (size_t x = 0; x < m_hiddenNeurons.size(); ++x)
			{
				for (size_t y = 0; y < m_hiddenNeurons[x].size(); ++y)
				{
					sf::Vector2f pos(hiddenOffset.x + deltaPos.x * x, hiddenOffset.y + deltaPos.y * y);
					m_hiddenNeurons[x][y]->size(m_neuronSize);
					m_hiddenNeurons[x][y]->pos(pos);
				}
			}

			for (size_t y = 0; y < m_outputNeurons.size(); ++y)
			{
				sf::Vector2f pos(outputOffset.x, outputOffset.y + deltaPos.y * y);
				m_outputNeurons[y]->size(m_neuronSize);
				m_outputNeurons[y]->pos(pos);
			}
		}
		
	};
};