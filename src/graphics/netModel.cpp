#include "graphics/netModel.h"

namespace NeuronalNet
{
	namespace Graphics
	{
		NetModel::NetModel(Net* net)
		{
			m_net				= net;
			m_connectionWidth	= ConnectionPainter::getStandardConnectionWidth();
			m_signalWidth		= ConnectionPainter::getStandardSignalWidth();
			m_neuronSize		= NeuronPainter::standardSize;
			m_neuronSpacing		= sf::Vector2f(50, 50);
			m_streamIndex = 0;
			build();
		}
		NetModel::~NetModel()
		{
			clear();
		}

		void NetModel::setStreamIndex(size_t index)
		{
			m_streamIndex = index;
			if (m_streamIndex >= m_net->getStreamSize())
				m_streamIndex = m_net->getStreamSize() - 1;


		}
		size_t NetModel::getStreamIndex() const
		{
			return m_streamIndex;
		}

		void NetModel::setPos(const sf::Vector2f& pos)
		{
			sf::Vector2f deltaPos = pos - m_pos;
			
			for (size_t i = 0; i < m_neuronList.size(); ++i)
				m_neuronList[i]->setPos(m_neuronList[i]->getPos() + deltaPos);
			m_pos = pos;
		}
		const sf::Vector2f& NetModel::getPos() const
		{
			return m_pos;
		}
		void NetModel::setNeuronSpacing(const sf::Vector2f& sp)
		{
			m_neuronSpacing = sp;
			updateNeuronDimensions();
		}
		const sf::Vector2f& NetModel::getNeuronSpacing()
		{
			return m_neuronSpacing;
		}

		void NetModel::build()
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
				NeuronPainter* neuron = DBG_NEW NeuronPainter();
				neuron->index(index);
				m_neuronList.push_back(neuron);
				m_inputNeurons.push_back(neuron);
			}

			for (size_t x = 0; x < m_net->getHiddenXCount(); ++x)
			{
				m_hiddenNeurons.push_back(vector<NeuronPainter*>());
				for (size_t y = 0; y < m_net->getHiddenYCount(); ++y)
				{
					NeuronIndex index;
					index.type = NeuronType::hidden;
					index.x = x;
					index.y = y;
					NeuronPainter* neuron = DBG_NEW NeuronPainter();
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
							ConnectionPainter* connection = DBG_NEW ConnectionPainter(m_inputNeurons[j],
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
							ConnectionPainter* connection = DBG_NEW ConnectionPainter(m_hiddenNeurons[x-1][j],
																	neuron);

							connection->index(conIndex);
							m_connectionList.push_back(connection);
						}
						//PixelPainter* pixPainter = DBG_NEW PixelPainter;
						//pixPainter->setDimenstions(m_net->getHiddenYCount(),
						//						   m_net->getHiddenYCount());
						//m_pixelPainterList.push_back(pixPainter);
					}

				}
				if (x == 0)
				{
					PixelPainter* pixPainter = DBG_NEW PixelPainter;
					pixPainter->setDimenstions(m_net->getInputCount()+1,
											   m_net->getHiddenYCount());
					m_pixelPainterList.push_back(pixPainter);
				}
				else
				{
					PixelPainter* pixPainter = DBG_NEW PixelPainter;
					pixPainter->setDimenstions(m_net->getHiddenYCount()+1,
											   m_net->getHiddenYCount());
					m_pixelPainterList.push_back(pixPainter);
				}
			}

			for (size_t y = 0; y < m_net->getOutputCount(); ++y)
			{
				NeuronIndex index;
				index.type = NeuronType::output;
				index.x = 0;
				index.y = y;
				NeuronPainter* neuron = DBG_NEW NeuronPainter();
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
						ConnectionPainter* connection = DBG_NEW ConnectionPainter(m_inputNeurons[j],
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
						ConnectionPainter* connection = DBG_NEW ConnectionPainter(m_hiddenNeurons[m_net->getHiddenXCount() - 1][j],
																neuron);

						connection->index(conIndex);
						m_connectionList.push_back(connection);
					}
				}
				
			}

			PixelPainter* pixPainter = DBG_NEW PixelPainter;
			if (m_net->getHiddenXCount() == 0)
			{
				pixPainter->setDimenstions(m_net->getInputCount()+1,
										   m_net->getOutputCount());
			}
			else
			{
				
				pixPainter->setDimenstions(m_net->getHiddenYCount()+1,
										   m_net->getOutputCount());
			}
			m_pixelPainterList.push_back(pixPainter);
			
			for (size_t i = 0; i < m_neuronList.size(); ++i)
				m_neuronInterface.push_back(m_neuronList[i]);
			for (size_t i = 0; i < m_connectionList.size(); ++i)
				m_connectionInterface.push_back(m_connectionList[i]);

			/*for (size_t i = 0; i < m_neuronList.size(); ++i)
				m_net->addGraphics(m_neuronList[i]);

			for (size_t i = 0; i < m_connectionList.size(); ++i)
				m_net->addGraphics(m_connectionList[i]);*/

			setConnectionWidth(m_connectionWidth);
			setSignalWidth(m_signalWidth);
			updateNeuronDimensions();
		}

		void NetModel::setConnectionWidth(float w)
		{
			m_connectionWidth = w;
			for (size_t i = 0; i < m_connectionList.size(); ++i)
				m_connectionList[i]->setConnectionWidth(m_connectionWidth);
		}
		void NetModel::setSignalWidth(float w)
		{
			m_signalWidth = w;
			for (size_t i = 0; i < m_connectionList.size(); ++i)
				m_connectionList[i]->setSignalWidth(m_signalWidth);
		}
		void NetModel::setNeuronSize(float size)
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


		float NetModel::getConnectionWidth() const
		{
			return m_connectionWidth;
		}
		float NetModel::getSignalWidth() const
		{
			return m_signalWidth;
		}
		float NetModel::getNeuronSize() const
		{
			return m_neuronSize;
		}

		void NetModel::draw(sf::RenderWindow* window,
							const sf::Vector2f &offset)
		{
			updateGraphics();

			for (size_t i = 0; i < m_connectionList.size(); ++i)
				m_connectionList[i]->draw(window, offset);
			for (size_t i = 0; i < m_neuronList.size(); ++i)
				m_neuronList[i]->draw(window, offset);
			for (size_t i = 0; i < m_pixelPainterList.size(); ++i)
				m_pixelPainterList[i]->draw(window, offset);
		}
		void NetModel::drawDebug(sf::RenderWindow* window,
								 const sf::Vector2f& offset)
		{

		}

		void NetModel::clear()
		{
			for (size_t i = 0; i < m_neuronList.size(); ++i)
			{
				//m_net->removeGraphics(m_neuronList[i]);
				delete m_neuronList[i];
			}
			m_neuronList.clear();

			for (size_t i = 0; i < m_connectionList.size(); ++i)
			{
				//m_net->removeGraphics(m_connectionList[i]);
				delete m_connectionList[i];
			}
			for (size_t i = 0; i < m_pixelPainterList.size(); ++i)
			{
				delete m_pixelPainterList[i];
			}
			m_pixelPainterList.clear();
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
				m_inputNeurons[y]->setSize(m_neuronSize);
				m_inputNeurons[y]->setPos(pos);
			}

			for (size_t x = 0; x < m_hiddenNeurons.size(); ++x)
			{
				sf::Vector2f pos;
				for (size_t y = 0; y < m_hiddenNeurons[x].size(); ++y)
				{
					pos = sf::Vector2f(hiddenOffset.x + deltaPos.x * x, hiddenOffset.y + deltaPos.y * y);
					m_hiddenNeurons[x][y]->setSize(m_neuronSize);
					m_hiddenNeurons[x][y]->setPos(pos);
				}
				pos.y = m_pos.y + deltaPos.y * maxYSize;
				sf::Vector2f offset;
				sf::Vector2f scale = m_pixelPainterList[x]->getDisplaySize();
				offset.y = -m_neuronSpacing.y/2.f;
				if (x == 0)
				{
					offset.x = -scale.x * (float)m_inputNeurons.size()/2.f;
				}
				else
				{
					offset.x = -scale.x * (float)m_hiddenNeurons[0].size() / 2.f;
				}
				offset.x -= deltaPos.x / 2.f;
				m_pixelPainterList[x]->setPos(pos + offset);
			}




			sf::Vector2f pos;
			for (size_t y = 0; y < m_outputNeurons.size(); ++y)
			{
				pos = sf::Vector2f(outputOffset.x, outputOffset.y + deltaPos.y * y);
				m_outputNeurons[y]->setSize(m_neuronSize);
				m_outputNeurons[y]->setPos(pos);
			}
			pos.y = m_pos.y + deltaPos.y * maxYSize;
			sf::Vector2f offset;
			sf::Vector2f scale = m_pixelPainterList[m_pixelPainterList.size() - 1]->getDisplaySize();
			offset.y = -m_neuronSpacing.y / 2.f;
			if (m_hiddenNeurons.size() == 0)
			{
				offset.x = -scale.x * (float)m_inputNeurons.size() / 2.f;
			}
			else
			{
				offset.x = -scale.x * (float)m_hiddenNeurons[0].size() / 2.f;
			}
			offset.x -= deltaPos.x / 2.f;
			m_pixelPainterList[m_pixelPainterList.size() - 1]->setPos(pos + offset);
		}
		void NetModel::updateGraphics()
		{
			m_net->graphics_update(m_neuronInterface, m_connectionInterface, m_streamIndex);

			const float* weight = m_net->getWeight();
			const float* bias   = m_net->getBias();
			size_t minElementW = getMinIndex<float>(weight, m_net->getWeightSize());
			size_t maxElementW = getMaxIndex<float>(weight, m_net->getWeightSize());

			size_t minElementB = getMinIndex<float>(bias, m_net->getNeuronCount());
			size_t maxElementB = getMaxIndex<float>(bias, m_net->getNeuronCount());

			float minW = weight[minElementW];
			float maxW = weight[maxElementW];
			float minB = bias[minElementB];
			float maxB = bias[maxElementB];

			float min = minW;
			float max = maxW;

			if (min > minB)
				min = minB;
			if (max > maxB)
				max = maxB;

			size_t wOffset = 0;
			size_t bOffset = 0;
			if (m_net->getHiddenXCount() == 0)
			{
				for (size_t inp = 0; inp < m_net->getInputCount()+1; ++inp)
				{
					if (inp == m_net->getInputCount())
					{
						// Bias
						for (size_t out = 0; out < m_net->getOutputCount(); ++out)
						{
							m_pixelPainterList[0]->setPixel(inp, out, getColor(bias[bOffset], min, max));
							++bOffset;
						}
					}
					else
					{
						// Weights
						for (size_t out = 0; out < m_net->getOutputCount(); ++out)
						{
							m_pixelPainterList[0]->setPixel(inp, out, getColor(weight[wOffset], min, max));
							++wOffset;
						}
					}
					
				}
			}
			else
			{
				for (size_t inp = 0; inp < m_net->getInputCount()+1; ++inp)
				{
					if (inp == m_net->getInputCount())
					{
						// Bias
						for (size_t hid = 0; hid < m_net->getHiddenYCount(); ++hid)
						{
							m_pixelPainterList[0]->setPixel(inp, hid, getColor(bias[bOffset], min, max));
							++bOffset;
						}
					}
					else
					{
						// Weights
						for (size_t hid = 0; hid < m_net->getHiddenYCount(); ++hid)
						{
							m_pixelPainterList[0]->setPixel(inp, hid, getColor(weight[wOffset], min, max));
							++wOffset;
						}
					}
					
				}
				for (size_t x = 1; x < m_net->getHiddenXCount(); ++x)
				{
					for (size_t hidI = 0; hidI < m_net->getHiddenYCount()+1; ++hidI)
					{
						if (hidI == m_net->getHiddenYCount())
						{
							// Bias
							for (size_t hidJ = 0; hidJ < m_net->getHiddenYCount(); ++hidJ)
							{
								m_pixelPainterList[x]->setPixel(hidI, hidJ, getColor(bias[bOffset], min, max));
								++bOffset;
							}
						}
						else
						{
							// Weights
							for (size_t hidJ = 0; hidJ < m_net->getHiddenYCount(); ++hidJ)
							{
								m_pixelPainterList[x]->setPixel(hidI, hidJ, getColor(weight[wOffset], min, max));
								++wOffset;
							}
						}
						
					}
				}
				for (size_t hidI = 0; hidI < m_net->getHiddenYCount()+1; ++hidI)
				{
					if (hidI == m_net->getHiddenYCount())
					{
						// Bias
						for (size_t out = 0; out < m_net->getOutputCount(); ++out)
						{
							m_pixelPainterList[m_pixelPainterList.size() - 1]->setPixel(hidI, out, getColor(bias[bOffset], min, max));
							++bOffset;
						}
					}
					else
					{
						// Weights
						for (size_t out = 0; out < m_net->getOutputCount(); ++out)
						{
							m_pixelPainterList[m_pixelPainterList.size() - 1]->setPixel(hidI, out, getColor(weight[wOffset], min, max));
							++wOffset;
						}
					}
					
				}
				
			}
		}
		
	};
};