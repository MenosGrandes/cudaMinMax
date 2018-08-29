CC = nvcc
CFLAGS =-std=c++11 
SRCS = main2.cu
PROG = minMax
BIN = bin
MKDIR_P = mkdir -p $(BIN)
RM = rm -rf



all : buildAll
debug :
	$(MKDIR_P)
	$(CC) $(CFLAGS) -DDEBUG -DSIZE=4 -DBLOCKSIZE=4 -o $(BIN)/$(PROG) $(SRCS) $(LIBS)

buildAll:$(SRCS)
	$(MKDIR_P)
	$(CC) $(CFLAGS) -DSIZE=1024 -DBLOCKSIZE=1024 -o $(BIN)/$(PROG) $(SRCS) $(LIBS)
clean:
	$(RM) $(BIN)

