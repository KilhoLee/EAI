CC := g++
CFLAGS := -std=c++11 -I/usr/include -I/usr/local/include/opencv4
INCLUDEPATHS := -I${HOME}/tensorflow -I${HOME}/EAI_TfLite 
LDFLAGS := -pthread -ltensorflow-lite -lflatbuffers -l:libedgetpu.so.1.0 -ldl\
					 -lopencv_core -lopencv_imgcodecs -lopencv_imgproc
					 
LDPATH := -L${HOME}/tensorflow/tensorflow/lite/tools/make/gen/bbb_armv7l/lib\
	  -L${HOME}/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/build\
		-L${HOME}/EAI/libs/armv7a

SRCS := tpu_latency.cc
OBJS := $(SRCS:.cc=.o)
EXEC := tpu_latency

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDEPATHS) -o $@ $^ $(LDPATH) $(LDFLAGS)  

%.o: %.cc
	$(CC) $(CFLAGS) $(INCLUDEPATHS) -c $< -o $@   

clean : 
	rm -f $(OBJS) $(EXEC)	
