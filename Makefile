#
# Makefile
#

CC = g++

CFLAGS = -I. -std=c++11 -pthread `pkg-config --cflags-only-I sdl`
LDFLAGS = `pkg-config --libs-only-L sdl`
LIBS = -lSOIL -lGLEW -lGLU -lglut `pkg-config --libs-only-l sdl`

OBJDIR = build/objects/
BINDIR = build/bin/

all: $(BINDIR)/convexhull

build:
	mkdir -p $(OBJDIR) $(BINDIR)

$(OBJDIR)/tiny_obj_loader.o: tinyobjloader/tiny_obj_loader.cc build
	$(CC) $(CFLAGS) -o $(OBJDIR)/tiny_obj_loader.o -c tinyobjloader/tiny_obj_loader.cc

$(OBJDIR)/camera.o: camera/camera.cpp build
	$(CC) $(CFLAGS) -o $(OBJDIR)/camera.o -c camera/camera.cpp

$(OBJDIR)/scene.o: scene/scene.cpp build
	$(CC) $(CFLAGS) -o $(OBJDIR)/scene.o -c scene/scene.cpp

$(OBJDIR)/input.o: input/input.cpp build
	$(CC) $(CFLAGS) -o $(OBJDIR)/input.o -c input/input.cpp

$(OBJDIR)/main.o: main.cpp build
	$(CC) $(CFLAGS) -o $(OBJDIR)/main.o -c main.cpp

$(BINDIR)/convexhull: $(OBJDIR)/tiny_obj_loader.o $(OBJDIR)/camera.o $(OBJDIR)/scene.o $(OBJDIR)/input.o $(OBJDIR)/main.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(BINDIR)/convexhull $(LIBS) $(OBJDIR)/tiny_obj_loader.o $(OBJDIR)/camera.o $(OBJDIR)/scene.o $(OBJDIR)/input.o $(OBJDIR)/main.o

clean:
	rm -rf build
