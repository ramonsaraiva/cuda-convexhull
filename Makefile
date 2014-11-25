#
# Makefile
#

CC = g++
NVCC = nvcc

CFLAGS = -I. -std=c++11 -pthread `pkg-config --cflags-only-I sdl`
CNFLAGS = -I. `pkg-config --cflags-only-I sdl`
NVCFLAGS = -arch sm_20 -I. `pkg-config --cflags-only-I sdl`

LDFLAGS = `pkg-config --libs-only-L sdl`
LIBS = -lSOIL -lGLEW -lGLU -lglut `pkg-config --libs-only-l sdl`

OBJDIR = build/objects/
BINDIR = build/bin/

all: $(BINDIR)/convexhull

cuda: $(BINDIR)/cuda

build:
	mkdir -p $(OBJDIR) $(BINDIR)

$(OBJDIR)/tiny_obj_loader.o: tinyobjloader/tiny_obj_loader.cc build
	$(CC) $(CNFLAGS) -o $(OBJDIR)/tiny_obj_loader.o -c tinyobjloader/tiny_obj_loader.cc

$(OBJDIR)/camera.o: camera/camera.cpp build
	$(CC) $(CNFLAGS) -o $(OBJDIR)/camera.o -c camera/camera.cpp

$(OBJDIR)/scene.o: scene/scene.cpp build
	$(CC) $(CNFLAGS) -o $(OBJDIR)/scene.o -c scene/scene.cpp

$(OBJDIR)/input.o: input/input.cpp build
	$(CC) $(CNFLAGS) -o $(OBJDIR)/input.o -c input/input.cpp

$(OBJDIR)/convexhull.o: convexhull/convexhull.cpp build
	$(CC) $(CFLAGS) -o $(OBJDIR)/convexhull.o -c convexhull/convexhull.cpp

$(OBJDIR)/main.o: main.cpp build
	$(CC) $(CFLAGS) -o $(OBJDIR)/main.o -c main.cpp

$(BINDIR)/convexhull: $(OBJDIR)/tiny_obj_loader.o $(OBJDIR)/camera.o $(OBJDIR)/scene.o $(OBJDIR)/input.o $(OBJDIR)/convexhull.o $(OBJDIR)/main.o
	$(CC) $(CNFLAGS) $(LDFLAGS) -o $(BINDIR)/convexhull $(LIBS) $(OBJDIR)/tiny_obj_loader.o $(OBJDIR)/camera.o $(OBJDIR)/scene.o $(OBJDIR)/input.o $(OBJDIR)/convexhull.o $(OBJDIR)/main.o

# CUDA

$(OBJDIR)/main.cu.o: main.cu build
	$(NVCC) $(NVCFLAGS) -o $(OBJDIR)/main.cu.o -c main.cu

$(BINDIR)/cuda: $(OBJDIR)/tiny_obj_loader.o $(OBJDIR)/camera.o $(OBJDIR)/scene.o $(OBJDIR)/input.o $(OBJDIR)/main.cu.o
	$(NVCC) $(NVCFLAGS) -o $(BINDIR)/cuda $(LIBS) $(OBJDIR)/tiny_obj_loader.o $(OBJDIR)/camera.o $(OBJDIR)/scene.o $(OBJDIR)/input.o $(OBJDIR)/main.cu.o

clean:
	rm -rf build
