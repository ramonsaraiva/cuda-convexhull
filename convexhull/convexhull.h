#ifndef CONVEXHULL
#define CONVEXHULL	1

#include <math.h>
#include <vector>
#include <map>
#include <stack>
#include <array>

#include <iostream>

typedef struct vec3_s
{
	float x;
	float y;
	float z;

		vec3_s(float _x, float _y, float _z)
	{
		x = _x;
		y = _y;
		z = _z;
	}

	vec3_s() {}

	vec3_s add(vec3_s& v)
	{
		struct vec3_s nv = vec3_s(x + v.x, y + v.y, z + v.z);
		return nv;
	}

	vec3_s sub(vec3_s& v)
	{
		struct vec3_s nv = vec3_s(x - v.x, y - v.y, z - v.z);
		return nv;
	}

	vec3_s mult(vec3_s& v)
	{
		struct vec3_s nv = vec3_s(x * v.x, y * v.y, z * v.z);
		return nv;
	}

	float dot(vec3_s& v)
	{
		return x * v.x + y * v.y + z * v.z;
	}

	vec3_s cross(vec3_s& v)
	{
		struct vec3_s c = vec3_s(y * v.z - z * v.y,
								 z * v.x - x * v.z,
								 x * v.y - y * v.x);
		return c;
	}

	vec3_s project_over(vec3_s& v)
	{
		float length = dot(v);
		struct vec3_s nv = vec3_s(length * v.x, length * v.y, length * v.z);
		return nv;
	}

	float length2()
	{
		return (x * x) + (y * y) + (z * z);
	}

	void normalize()
	{
		float length = length2();
		if (length == 0)
			return;

		length = sqrt(length);
		x = x/length;
		y = y/length;
		z = z/length;
	}

	void debug()
	{
		std::cout << x << " # " << y << " # " << z << std::endl;
	}
} vec3;

void sanitize(std::vector<vec3>& points);
int giftwrap(std::vector<vec3>& points, std::vector<std::array<vec3, 3>>& polys);
int lower(std::vector<vec3> points);
int next_point(std::vector<vec3> points, int p1i, int p2i);
bool edge_exists(std::map<std::string, bool>& created, int p1, int p2);
void add_edge(std::map<std::string, bool>& created, std::stack<std::array<int, 2>>& edges, int p1, int p2);

//tests
void vec3_all_tests();

bool vec3_add_test();
bool vec3_sub_test();
bool vec3_cross_test();
bool vec3_aa_normalize_test();
bool vec3_length1_normalize_test();
bool vec3_po_axis_test();

void ch_all_tests();
void ch_simple_lower_test();
void ch_lower_with_tie_test();
void ch_lower_point_bug_case_test();
void ch_tetrahedron_test();



#endif
