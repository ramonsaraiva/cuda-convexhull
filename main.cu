#include <stdio.h>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>
#include <stack>
#include <map>

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
std::vector< std::vector<vec3> > polys;


/*
   CUDA DATA
*/

cuvec3* in_points;
int points_size;

int* out_points;
int* aux_points;
int* cu_points_size;
int* next_p;
int* created_edges;

/*
   END CUDA DATA
*/

void* setup_sdl();
void setup_gl();
void giftwrap();
void render();

void sanitize(std::vector<vec3>& points);
bool edge_exists(std::map<std::string, bool>& created, int p1, int p2);
void add_edge(std::map<std::string, bool>& created, std::stack< std::vector<int> >& edges, int p1, int p2);

/*
   CUDA FUNCTIONS
*/

int err(cudaError_t s);

__global__ void test();
__global__ void lower(int* cu_points_size, cuvec3* in_points, int* out_points);
__global__ void next_point(int* cu_points_size, cuvec3* in_points, int* out_points, int* aux_points, int* next_p, int p1_i, int p2_i);

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

	giftwrap();

	input_ctr = InputController();
	while (1)
	{
		input_ctr.events();
		render();
	}

	return 0;
}

void giftwrap()
{
	cudaError_t cuda_s;
	cudaEvent_t start, stop;
	float time_ms;

	int p1;
	int p2;

	int blocks;
	int threads;

	std::stack< std::vector<int> > open_edges;
	std::map<std::string, bool> created_edges;

	cuda_s = cudaSetDevice(0);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cuda_s = cudaMalloc((void**) &cu_points_size, sizeof(int));
	cuda_s = cudaMalloc((void**) &in_points, points_size * sizeof(cuvec3));
	cuda_s = cudaMalloc((void**) &out_points, points_size * sizeof(int));
	cuda_s = cudaMalloc((void**) &aux_points, points_size * sizeof(int));
	cuda_s = cudaMalloc((void**) &next_p, sizeof(int));
	cuda_s = cudaMalloc((void**) &created_edges, 2 * points_size * sizeof(int));

	cuda_s = cudaMemcpy(cu_points_size, &points_size, sizeof(int), cudaMemcpyHostToDevice);
	cuda_s = cudaMemcpy(in_points, points, points_size * sizeof(cuvec3), cudaMemcpyHostToDevice);

	if (points_size > 1024)
	{
		threads = 1024;
		blocks = ceil(points_size / 1024);
	}
	else
	{
		blocks = 1;
		threads = points_size;
	}

	cudaEventRecord(start);

	lower<<<blocks, threads>>>(cu_points_size, in_points, out_points);
	next_point<<<blocks, threads>>>(cu_points_size, in_points, out_points, aux_points, next_p, -1, -1);

	cuda_s = cudaDeviceSynchronize();

	cuda_s = cudaMemcpy(&p1, out_points, sizeof(int), cudaMemcpyDeviceToHost);
	cuda_s = cudaMemcpy(&p2, next_p, sizeof(int), cudaMemcpyDeviceToHost);

	add_edge(created_edges, open_edges, p2, p1);

	while (!open_edges.empty())
	{
		p1 = open_edges.top()[0];
		p2 = open_edges.top()[1];
		open_edges.pop();


		if (edge_exists(created_edges, p1, p2))
			continue;

		int p3;

		next_point<<<blocks, threads>>>(cu_points_size, in_points, out_points, aux_points, next_p, p1, p2);
		cuda_s = cudaDeviceSynchronize();
		cuda_s = cudaMemcpy(&p3, next_p, sizeof(int), cudaMemcpyDeviceToHost);

		std::vector<vec3> v;
		v.push_back(points[p1]);
		v.push_back(points[p2]);
		v.push_back(points[p3]);

		polys.push_back(v);

		add_edge(created_edges, open_edges, p1, p2);
		add_edge(created_edges, open_edges, p2, p3);
		add_edge(created_edges, open_edges, p3, p1);
	}

	cuda_s = cudaDeviceSynchronize();
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	time_ms = 0;
	cudaEventElapsedTime(&time_ms, start, stop);

	cuda_s = cudaDeviceReset();

	printf("polys: %d\n", polys.size());
	printf("time %f\n", time_ms/1000.0);

	cudaFree(cu_points_size);
	cudaFree(in_points);
	cudaFree(out_points);
	cudaFree(aux_points);
	cudaFree(next_p);
}

__global__ void next_point(int* cu_points_size, cuvec3* in_points, int* out_points, int* aux_points, int* next_p, int p1_i, int p2_i)
{
	unsigned int tid = threadIdx.x;
	unsigned int p = blockIdx.x * blockDim.x + tid;

	int candidate_i = p * POINTS_PER_THREAD;

	cuvec3 p1;
	cuvec3 p2;
	cuvec3 edge;

	if (candidate_i < *cu_points_size)
	{
		if (p1_i < 0)
			p1 = in_points[out_points[0]];
		else
			p1 = in_points[p1_i];

		if (p2_i < 0)
		{
			cuvec3 v = cuvec3(1, 1, 0);
			p2 = p1.sub(v);
		}
		else
		{
			p2 = in_points[p2_i];
		}

		edge = p2.sub(p1);
		edge.normalize();

		for (int i = p * POINTS_PER_THREAD + 1; i < (p + 1) * POINTS_PER_THREAD && i < *cu_points_size; i++)
		{
			if (i == p1_i || i == p2_i)
				continue;

			cuvec3 v = in_points[i].sub(p1);
			cuvec3 po = v.project_over(edge);
			v = v.sub(po);

			cuvec3 candidate = in_points[candidate_i].sub(p1);
			po = candidate.project_over(edge);
			candidate = candidate.sub(po);

			cuvec3 cross = v.cross(candidate);
			if (cross.dot(edge) > 0)
				candidate_i = i;
		}

		aux_points[p] = candidate_i;
	}

	__syncthreads();

	if (p == 0)
	{
		candidate_i = -1;

		int div = (*cu_points_size % POINTS_PER_THREAD == 0) ? *cu_points_size / POINTS_PER_THREAD : (*cu_points_size / POINTS_PER_THREAD) + 1;
		for (int i = 0; i < div; i++)
		{
			if (aux_points[i] == p1_i || aux_points[i] == p2_i)
				continue;

			if (candidate_i == -1)
			{
				candidate_i = i;
				continue;
			}

			cuvec3 v = in_points[aux_points[i]].sub(p1);
			cuvec3 po = v.project_over(edge);
			v = v.sub(po);

			cuvec3 candidate = in_points[aux_points[candidate_i]].sub(p1);
			po = candidate.project_over(edge);
			candidate = candidate.sub(po);

			cuvec3 cross = v.cross(candidate);
			if (cross.dot(edge) > 0)
			{
				candidate_i = i;
			}
		}

		*next_p = aux_points[candidate_i];
	}
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
		low = out_points[0];

		int div = (*cu_points_size % POINTS_PER_THREAD == 0) ? *cu_points_size / POINTS_PER_THREAD : (*cu_points_size / POINTS_PER_THREAD) + 1;
		for (int i = 0; i < div; i++)
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

		out_points[0] = out_points[low];
	}
}


void setup_gl()
{
	glCullFace(GL_BACK);
	glFrontFace(GL_CW);
	glEnable(GL_CULL_FACE);

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

	glColor3f(1.0, 1.0, 1.0);
	glBegin(GL_TRIANGLES);
	for (int i = 0; i < polys.size(); i++)
	{
		vec3 v1 = polys[i][1].sub(polys[i][0]);
		vec3 v2 = polys[i][2].sub(polys[i][0]);
		vec3 normal = v2.cross(v1);
		normal.normalize();

		glNormal3f(normal.x, normal.y, normal.z);

		for (int j = 0; j < 3; j++)
		{
			glVertex3f(polys[i][j].x, polys[i][j].y, polys[i][j].z);
		}
	}
	glEnd();

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
				points.erase(points.begin() + j);
				--j;
				--i;
			}
		}
	}
}

bool edge_exists(std::map<std::string, bool>& created, int p1, int p2)
{
	char key_c[32];
	std::string key;

	sprintf(key_c, "%d_%d", p1, p2);
	key = std::string(key_c);

	return created[key];
}

void add_edge(std::map<std::string, bool>& created, std::stack< std::vector<int> >& edges, int p1, int p2)
{
	char key_c[32];
	std::string key;

	sprintf(key_c, "%d_%d", p1, p2);
	key = std::string(key_c);

	created[key] = true;

	if (!edge_exists(created, p2, p1))
	{
		std::vector<int> edge;
		edge.push_back(p2);
		edge.push_back(p1);

		edges.push(edge);
	}
}

int err(cudaError_t s)
{
	if (s == cudaSuccess)
		return 0;

	fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(s));

	return 1;
}
