#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>


using std::string;
using std::vector;

enum Label
{
	zero,
	one,
	two,
	three,
	four,
	five,
	six,
	seven,
	eight,
	nine,

	count
};

class ImageGray
{
	public:
	ImageGray();
	ImageGray(const vector<float>& pixelsVector, size_t xDim, size_t yDim);
	~ImageGray();

	void setPixels(const vector<float>& pixelsVector, size_t xDim, size_t yDim);
	void setPixel(size_t x, size_t y, float brightness);
	void setLabel(Label label);


	const vector<float>& getPixels() const;
	float getPixel(size_t x, size_t y) const;
	Label getLabel() const;
	const vector<float>& getLabelVector() const;

	private:
	inline size_t getIndex(size_t x, size_t y) const;
	vector<float>  m_pixelsVector; // Inputs
	vector<float>  m_labelVector;  // Desired output
	size_t m_xDim;
	size_t m_yDim;
	Label m_label;
};
class Dataset
{
	public:
	
	Dataset(int scaleDividor = 1);
	~Dataset();

	void clear();
	bool importBinaryData(const string& filePath, size_t pixelX, size_t pixelY, Label label);
	void shuffle();
	const vector<ImageGray>& getImages() const;
	const ImageGray &getImage(size_t index);
	size_t count() const;

	static string labelToString(Label l);
	//sf::Sprite getImageSprite(size_t index);

	//bool importImage(const string& folderPath);

	private:

	vector<ImageGray> m_images;
	vector<size_t> m_shuffeledIndex;
	int m_scale;
};