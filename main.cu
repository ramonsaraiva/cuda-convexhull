#include <stdio.h>
#include <iostream>
#include <math.h>

#include <SDL/SDL.h>
#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include "tinyobjloader/tiny_obj_loader.h"

#include "thrust/device_vector.h"

#include "camera/camera.h"
#include "scene/scene.h"
#include "input/input.h"

#define WIDTH 900
#define HEIGHT 700

#define POINTS_PER_THREAD	2

typedef struct cuvec3_s
{
	float x;
	float y;
	float z;

	__device__ cuvec3_s(float _x, float _y, float _z)
	{
		x = _x;
		y = _y;
		z = _z;
	}

	__device__ cuvec3_s() {}

	__device__ cuvec3_s add(cuvec3_s& v)
	{
		struct cuvec3_s nv = cuvec3_s(x + v.x, y + v.y, z + v.z);
		return nv;
	}

	__device__ cuvec3_s sub(cuvec3_s& v)
	{
		struct cuvec3_s nv = cuvec3_s(x - v.x, y - v.y, z - v.z);
		return nv;
	}

	__device__ cuvec3_s mult(cuvec3_s& v)
	{
		struct cuvec3_s nv = cuvec3_s(x * v.x, y * v.y, z * v.z);
		return nv;
	}

	__device__ float dot(cuvec3_s& v)
	{
		return x * v.x + y * v.y + z * v.z;
	}

	__device__ cuvec3_s cross(cuvec3_s& v)
	{
		struct cuvec3_s c = cuvec3_s(y * v.z - z * v.y,
								 z * v.x - x * v.z,
								 x * v.y - y * v.x);
		return c;
	}

	__device__ cuvec3_s project_over(cuvec3_s& v)
	{
		float length = dot(v);
		struct cuvec3_s nv = cuvec3_s(length * v.x, length * v.y, length * v.z);
		return nv;
	}

	__device__ float length2()
	{
		return (x * x) + (y * y) + (z * z);
	}

	__device__ void normalize()
	{
		float length = length2();
		if (length == 0)
			return;

		length = sqrt(length);
		x = x/length;
		y = y/length;
		z = z/length;
	}

	__device__ void debug()
	{
		printf("%f # %f # %f\n", x, y, z);
	}
} cuvec3;

InputController input_ctr;

std::vector<vec3> obj_points;
vec3* points;

/*
   CUDA DATA
*/

cuvec3* in_points;
int* out_points;
int* cu_points_size;
int points_size;
std::vector<int> threads;

thrust::device_vector<int> dvec(10, 5);

/*
   END CUDA DATA
*/

void* setup_sdl();
void setup_gl();
void setup_cuda();
void render();

void sanitize(std::vector<vec3>& points);

/*
   CUDA FUNCTIONS
*/

__global__ void test(int* ptr);
__global__ void lower(int* cu_points_size, cuvec3* in_points, int* out_points);

/*
   END CUDA FUNCTIONS
*/

int main(int argc, char** argv)
{
	setup_sdl();
	glewInit();
	setup_gl();
	glutInit(&argc, argv);

	Camera cam = Camera(90, WIDTH, HEIGHT, 1000);
	Scene::instance().set_default_camera(&cam);

	SceneObject obj = SceneObject("obj");
	obj.load_obj(std::string("primitives/" + std::string(argv[1]) + "/" + std::string(argv[1]) + ".obj").c_str());
	obj.build_vbo();
	obj.set_render_mode(GL_TRIANGLES);

	Scene::instance().add_object("obj", &obj);

	obj.points(obj_points);
	sanitize(obj_points);

	points_size = obj_points.size();
	std::cout << "points size: " << points_size << std::endl;

	points = (vec3*) malloc(points_size * sizeof(vec3));

	for (int i = 0; i < obj_points.size(); i++)
	{
		points[i] = obj_points[i];
	}

	setup_cuda();

	input_ctr = InputController();
	while (1)
	{
		input_ctr.events();
		render();
	}

	return 0;
}

__global__ void test(int* ptr)
{
	printf("emdcwqklfm,çlwqrçlqe %d\n", ptr[0]);
}

__global__ void lower(int* cu_points_size, cuvec3* in_points, int* out_points)
{
	unsigned int tid = threadIdx.x;
	unsigned int p = blockIdx.x * blockDim.x + tid;

	int low = p * POINTS_PER_THREAD;

	if (low < *cu_points_size)
	{
		for (int i = p * POINTS_PER_THREAD; i < (p + 1) * POINTS_PER_THREAD && i < *cu_points_size; i++)
		{
			if (in_points[i].z < in_points[low].z)
			{
				low = i;
			}
			else if (fabs(in_points[i].z - in_points[low].z) < 0.002)
			{
				if (in_points[i].y < in_points[low].y)
				{
					low = i;
				}
				else if (fabs(in_points[i].y - in_points[low].y) < 0.002)
				{
					if (in_points[i].x < in_points[low].x)
						low = i;
				}
			}
		}

		out_points[p] = low;
	}

	__syncthreads();

	if (p == 0)
	{
		low = 0;

		for (int i = 0; i < *cu_points_size / POINTS_PER_THREAD; i++)
		{
			if (in_points[out_points[i]].z < in_points[out_points[low]].z)
			{
				low = i;
			}
			else if (fabs(in_points[out_points[i]].z - in_points[out_points[low]].z) < 0.002)
			{
				if (in_points[out_points[i]].y < in_points[out_points[low]].y)
				{
					low = i;
				}
				else if (fabs(in_points[out_points[i]].y - in_points[out_points[low]].y) < 0.002)
				{
					if (in_points[out_points[i]].x < in_points[out_points[low]].x)
						low = i;
				}
			}
		}

		printf("low => %d\n", out_points[low]);
		out_points[0] = out_points[low];
	}
}

void setup_cuda()
{
	cudaError_t cuda_s;

	cuda_s = cudaSetDevice(0);
	cuda_s = cudaMalloc((void**) &cu_points_size, sizeof(int));
	cuda_s = cudaMalloc((void**) &in_points, points_size * sizeof(cuvec3));
	cuda_s = cudaMalloc((void**) &out_points, points_size * sizeof(cuvec3));

	cuda_s = cudaMemcpy(cu_points_size, &points_size, sizeof(int), cudaMemcpyHostToDevice);
	cuda_s = cudaMemcpy(in_points, points, points_size * sizeof(cuvec3), cudaMemcpyHostToDevice);

	lower<<<2, 181>>>(cu_points_size, in_points, out_points);

	int* ptr = thrust::raw_pointer_cast(&dvec[0]);

	thrust::device_vector<int> open_edges (points_size * 2 * 2, 0);

	test<<<1, 1>>>(ptr);

	cuda_s = cudaDeviceSynchronize();

	cuda_s = cudaMemcpy(points, out_points, points_size * sizeof(cuvec3), cudaMemcpyDeviceToHost);

	cuda_s = cudaDeviceReset();

	/*
	for (int i = 0; i < obj_points.size(); i++)
	{
	}
	*/

	cudaFree(in_points);
	cudaFree(out_points);
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
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHTING);
	/*
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

	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, ambient_light);

	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);

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

	/*
	glColor3f(1.0, 1.0, 1.0);
	glBegin(GL_TRIANGLES);
	for (int i = 0; i < polys.size(); i++)
	{
		vec3 v1 = polys[i][1].sub(polys[i][0]);
		vec3 v2 = polys[i][2].sub(polys[i][0]);
		vec3 normal = v1.cross(v2);
		normal.normalize();

		glNormal3f(normal.x, normal.y, normal.z);

		for (int j = 0; j < 3; j++)
		{
			glVertex3f(polys[i][j].x, polys[i][j].y, polys[i][j].z);
		}
	}
	glEnd();
	*/

	SDL_GL_SwapBuffers();
}

void sanitize(std::vector<vec3>& points)
{
	std::map<int, bool> to_delete;

	for (int i = 0; i < points.size(); i++)
	{
		for (int j = i+1; j < points.size(); j++)
		{
			if (fabs(points[i].x - points[j].x) < 0.002 &&
				fabs(points[i].y - points[j].y) < 0.002 &&
				fabs(points[i].z - points[j].z) < 0.002)
			{
				to_delete[j] = true;
			}
		}
	}

	for(std::map<int, bool>::iterator iter = to_delete.begin(); iter != to_delete.end(); iter++)
	{
		points.erase(points.begin() + iter->first);
	}
}
