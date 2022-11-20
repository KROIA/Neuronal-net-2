DEFINES += DLL_EXPORT
include(CUDA.pri)
include(Neural-net-2.pri)


TARGET = Neural-net-2
TEMPLATE = lib
LIBS += -L$$PWD/bin/Release/cuda -llibNeural-net-2


