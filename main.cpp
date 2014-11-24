#include <iostream>

#include <SDL/SDL.h>
#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include "tinyobjloader/tiny_obj_loader.h"

#include "camera/camera.h"
#include "scene/scene.h"
#include "input/input.h"
#include "convexhull/convexhull.h"

#define WIDTH 900
#define HEIGHT 700

using namespace std;

InputController input_ctr;

void* setup_sdl();
void setup_gl();
void render();

std::vector<vec3> cube_points;
std::vector<int> polys;

int main(int argc, char** argv)
{
	setup_sdl();
	glewInit();
	setup_gl();
	glutInit(&argc, argv);

	Camera cam = Camera(90, WIDTH, HEIGHT, 1000);
	Scene::instance().set_default_camera(&cam);

	SceneObject cube = SceneObject("cube");
	cube.load_obj("primitives/sphere/sphere.obj");
	cube.build_vbo();
	cube.set_render_mode(GL_POINTS);

	Scene::instance().add_object("cube", &cube);

	//hull
	cube.points(cube_points);
	sanitize(cube_points);

	giftwrap(cube_points, polys);

	input_ctr = InputController();
	while (1)
	{
		input_ctr.events();
		render();
	}

	return 0;
}

void setup_gl()
{
	/*
	   glCullFace(GL_BACK);
	   glFrontFace(GL_CCW);
	   glEnable(GL_CULL_FACE);
	*/

	glClearColor(0, 0, 0, 0);
	glClearDepth(1.0);
	/*
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHTING);
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_COLOR_MATERIAL);
	*/

	glDepthFunc(GL_LESS);
	glEnable(GL_DEPTH_TEST);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glViewport(0, 0, WIDTH, HEIGHT);

	GLfloat light_position[] = { 1.0, 1.0, 1.0, 0.0 };
	GLfloat light_specular[] = {1.0, 1.0, 1.0,1.0};
	GLfloat light_diffuse[] = {1.0, 1.0, 1.0,1.0};
	GLfloat ambient_light[] = {0.5f, 0.5f, 0.5f, 1.0f};

	glShadeModel(GL_SMOOTH);

	/*
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient_light);

	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	*/

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glPointSize(10);
	gluPerspective(60.0, WIDTH / HEIGHT, 1.0, 1024.0);
}

void* setup_sdl()
{
	const SDL_VideoInfo* vinfo = NULL;
	SDL_Surface* screen;

	int bpp = 0;
	int flags = 0;

	if (SDL_Init(SDL_INIT_EVERYTHING) < 0)
		exit(1);

	vinfo = SDL_GetVideoInfo();

	if (!vinfo)
		exit(1);

	SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 5);
	SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 5);
	SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 5);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

	bpp = vinfo->vfmt->BitsPerPixel;
	flags = SDL_OPENGL;

	if ((screen = SDL_SetVideoMode(WIDTH, HEIGHT, bpp, flags)) == 0)
		exit(1);

	SDL_WM_SetCaption("ObjAnim", NULL);

	return screen;
}

void render()
{
	glClearDepth(1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	Scene::instance().default_camera()->refresh_lookat();
	//Scene::instance().render();

	glPushMatrix();
	glColor3f(1.0, 0.5, 0.5);
	glBegin(GL_LINES);
	for (int i = 0; i < polys.size(); i++)
	{
		glVertex3f(cube_points[polys[i]].x, cube_points[polys[i]].y, cube_points[polys[i]].z);
	}
	glEnd();
	glPopMatrix();

	SDL_GL_SwapBuffers( );
}
