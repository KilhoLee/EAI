CC := g++
CFLAGS := -std=c++11 -I/usr/include
INCLUDEPATHS := -I${HOME}/tensorflow
LDFLAGS := -pthread -ltensorflow-lite -lflatbuffers -ldl
LDPATH := -L${HOME}/tensorflow/tensorflow/lite/tools/make/gen/bbb_armv7l/lib\
	  -L${HOME}/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build

SRCS := tflite_sample.cc
OBJS := $(SRCS:.cc=.o)
EXEC := tflite_sample

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDEPATHS) -o $@ $^ $(LDPATH) $(LDFLAGS)  

%.o: %.cc
	$(CC) $(CFLAGS) $(INCLUDEPATHS) -c $< -o $@   

clean : 
	rm -f $(OBJS) $(EXEC)
