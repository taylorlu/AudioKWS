
FFTW = ../third_party/fftw-3.3.4/build/

OPT = -O2

specgram:
	g++ --std=c++11 $(OPT) -c specgram.cpp \
		-fPIC -I$(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE) -I$(BOOST_INCLUDE) \
		-I../third_party/ -I$(FFTW)/include
	g++ -std=c++11 $(OPT) -shared -rdynamic specgram.o \
	   	-L$(BOOST_LIB) -lboost_python \
		-L/usr/lib/python$(PYTHON_VERSION)/config \
		-lpython$(PYTHON_VERSION) -Wl,-rpath `pwd` \
		-L$(FFTW)/lib -lfftw3f -Wl,-rpath $(FFTW)/lib \
		-o cspecgram.so

clean:
	rm -f *.o *.so
