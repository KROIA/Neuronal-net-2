#include "dataset.h"

ImageGray::ImageGray()
{
	m_labelVector.resize(Label::count, -1);
}
ImageGray::ImageGray(const vector<float>& pixelsVector, size_t xDim, size_t yDim)
{
	m_labelVector.resize(Label::count, -1);
	setPixels(pixelsVector, xDim, yDim);
}
ImageGray::~ImageGray()
{

}

void ImageGray::setPixels(const vector<float>& pixelsVector, size_t xDim, size_t yDim)
{
	m_pixelsVector = pixelsVector;
	while (m_pixelsVector.size() < xDim * yDim)
		m_pixelsVector.push_back(0);

	m_xDim = xDim;
	m_yDim = yDim;
}

void ImageGray::setPixel(size_t x, size_t y, float brightness)
{
	if (m_xDim <= x || m_yDim <= y)
		return;
	m_pixelsVector[getIndex(x, y)] = brightness;
}
void ImageGray::setLabel(Label label)
{
	if (label > Label::count)
	{
		std::cout << "Label out of range\n";
		return;
	}
	m_label = label;

	m_labelVector = vector<float>(Label::count, -0.5);
	//memset(m_labelVector.data(), -1, Label::count * sizeof(float));
	m_labelVector[m_label] = 0.5;
}

const vector<float>& ImageGray::getPixels() const
{
	return m_pixelsVector;
}
float ImageGray::getPixel(size_t x, size_t y) const
{
	if (m_xDim <= x || m_yDim <= y)
		return 0;

	return m_pixelsVector[getIndex(x, y)];
}
Label ImageGray::getLabel() const
{
	return m_label;
}
const vector<float>& ImageGray::getLabelVector() const
{
	return m_labelVector;
}
inline size_t ImageGray::getIndex(size_t x, size_t y) const
{
	return x + m_xDim * y;
}





Dataset::Dataset(int scaleDividor)
{

	m_scale = scaleDividor;
	if (m_scale < 1)
		m_scale = 1;
}
Dataset::~Dataset()
{

}

void Dataset::clear()
{
	m_images.clear();
	m_shuffeledIndex.clear();
}
bool Dataset::importBinaryData(const string& filePath, size_t pixelX, size_t pixelY, Label label)
{
	std::ifstream file;
	file.open(filePath, std::ios::binary);
	if (!file.is_open())
	{
		std::cout << "Can't open file: \"" << filePath << "\"\n";
		return false;
	}
		

	// copies all data into buffer
	std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(file), {});

	file.close();

	// Parse data
	size_t imageBufferSize = pixelX * pixelY;
	size_t imageIndex = 0;
	size_t bufferIndex = 0;
	bool parse = true;
	while(parse)
	{
		if (buffer.size() < bufferIndex + imageBufferSize)
			break;

		if (m_scale == 1)
		{
			vector<float>  pixelsVector(imageBufferSize, 0);
			for (size_t i = 0; i < imageBufferSize; ++i)
			{
				pixelsVector[i] = (float)buffer[bufferIndex + i] / 128.f - 1.f;
				if (pixelsVector[i] > 1 || pixelsVector[i] < -1)
				{
					std::cout << "pixel: " << pixelsVector[i];
				}
			}
			bufferIndex += imageBufferSize;

			m_shuffeledIndex.push_back(m_images.size());
			m_images.push_back(ImageGray(pixelsVector, pixelX, pixelY));
		}
		else
		{
			size_t buffSize = imageBufferSize / m_scale;
			vector<float>  pixelsVector(imageBufferSize, 0);
			
			for (size_t i = 0; i < imageBufferSize; ++i)
			{
				pixelsVector[i] = (float)buffer[bufferIndex + i] / 128.f - 1.f;
				if (pixelsVector[i] > 1 || pixelsVector[i] < -1)
				{
					std::cout << "pixel: " << pixelsVector[i];
				}
			}
			bufferIndex += imageBufferSize;

			ImageGray tmpImage(pixelsVector, pixelX, pixelY);
			vector<float>  pixelsVector2(imageBufferSize/ m_scale, 0);

			for (size_t x = 0; x < pixelX / m_scale; ++x)
			{
				for (size_t y = 0; y < pixelY / m_scale; ++y)
				{
					float pixVal = 0;
					for (size_t x1 = 0; x1 < m_scale; ++x1)
					{
						for (size_t y1 = 0; y1 < m_scale; ++y1)
						{
							pixVal += tmpImage.getPixel((x*m_scale) + x1, (y*m_scale) + y1);
						}
					}
					pixVal /= (float)(m_scale * m_scale);
					pixelsVector2[y * (pixelX / m_scale) + x] = pixVal;
				}
			}
			m_shuffeledIndex.push_back(m_images.size());
			m_images.push_back(ImageGray(pixelsVector2, pixelX / m_scale, pixelY / m_scale));
		}
		m_images[m_images.size() - 1].setLabel(label);
		
	}
	std::cout << m_images.size() << " Images imported \n";
	return true;
}
/*bool Dataset::importImage(const string& folderPath)
{

}*/
void Dataset::shuffle()
{
	auto rng = std::default_random_engine{};
	std::shuffle(std::begin(m_shuffeledIndex), std::end(m_shuffeledIndex), rng);
}

const vector<ImageGray>& Dataset::getImages() const
{
	return m_images;
}
const ImageGray& Dataset::getImage(size_t index)
{
	if (index >= m_images.size())
	{
		std::cout << "Dataset::getImage(size_t index) index out of range";
		return m_images[0];
	}
	return m_images[m_shuffeledIndex[index]];
}

size_t Dataset::count() const
{
	return m_images.size();
}

string Dataset::labelToString(Label l)
{
	switch (l)
	{
		case zero:
			return "zero";
		case one:
			return "one";
		case two:
			return "two";
		case three:
			return "three";
		case four:
			return "four";
		case five:
			return "five";
		case six:
			return "six";
		case seven:
			return "seven";
		case eight:
			return "eight";
		case nine:
			return "nine";
		default:
			return "not in dataset";
	}
	return "not in dataset";
}

/*sf::Sprite Dataset::getImageSprite(size_t index)
{
	ImageGray& image = m_images[index];

}*/