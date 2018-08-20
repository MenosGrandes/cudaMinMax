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
	$(CC) $(CFLAGS) -DDEBUG -DSIZE=16 -DBLOCKSIZE=16 -o $(BIN)/$(PROG) $(SRCS) $(LIBS)

buildAll:$(SRCS)
	$(MKDIR_P)
	$(CC) $(CFLAGS) -DSIZE=16 -DBLOCKSIZE=16 -o $(BIN)/$(PROG) $(SRCS) $(LIBS)
clean:
	$(RM) $(BIN)

