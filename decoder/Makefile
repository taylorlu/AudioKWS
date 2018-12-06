
all: kws.so

OPT = -O2

kws.so:
	g++ -std=c++11 $(OPT) -I$(PYTHON_INCLUDE) -I$(BOOST_INCLUDE) -I$(NUMPY_INCLUDE) \
		 -fPIC -c kws.cpp

	g++ -std=c++11 $(OPT) -shared -rdynamic kws.o -L$(BOOST_LIB) \
		-lboost_python -L/usr/local/lib \
		-lpython$(PYTHON_VERSION) -o kws.so

clean:
	rm -f *.o *.so
