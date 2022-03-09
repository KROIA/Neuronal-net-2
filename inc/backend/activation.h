#pragma once
#include <math.h>

typedef enum class Activation
{
	linear,
	finiteLinear,
	binary,
	sigmoid,
	gauss
};

#define NET_ACTIVATION_LINEAR(x) x
#define NET_ACTIVATION_LINEAR_DERIVETIVE(x) 1

#define NET_ACTIVATION_RELU(x) x<=0? 0:x
#define NET_ACTIVATION_RELU_DERIVETIVE(x) x<=0? 0:1

#define NET_ACTIVATION_FINITELINEAR(x) x<-1? -1: x>1? 1:x
#define NET_ACTIVATION_FINITELINEAR_DERIVETIVE(x) x<-1? 0: x>1? 0:1

#define NET_ACTIVATION_BINARY(x) x<0? -1:1
// No derivetive exists

//https://www.wolframalpha.com/input/?i=exp%28-pow%28x%2C2%29%29*2-1
#define NET_ACTIVATION_GAUSSIAN(x) 2*pow(2.71828182845904523536f,-pow((float)x,2))-1 
#define NET_ACTIVATION_GAUSSIAN_DERIVETIVE(x) -4*pow(2.71828182845904523536f,-pow(x,2))*x 


#define NET_ACTIVATION_SIGMOID(x) atan((float)x)* 0.628318531f
#define NET_ACTIVATION_SIGMOID_DERIVETIVE(x) 0.628318531f/(pow(x,2)+1)