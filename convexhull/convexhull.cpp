#include <vector>
#include <iostream>
#include <string>
#include <math.h>
#include <map>
#include <stack>
#include <array>

#include "convexhull.h"

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

bool edge_exists(std::map<std::string, bool>& created, int p1, int p2)
{
	std::string key = std::to_string(p1) + "_" + std::to_string(p2);
	return created[key];
}

void add_edge(std::map<std::string, bool>& created, std::stack<std::array<int, 2>>& edges, int p1, int p2)
{
	std::string key = std::to_string(p1) + "_" + std::to_string(p2);
	created[key] = true;

	if (!edge_exists(created, p2, p1))
	{
		std::array<int, 2> e = {p2, p1};
		edges.push(e);
	}
}

int giftwrap(std::vector<vec3>& points, std::vector<int>& polys)
{
	std::stack<std::array<int, 2>> open_edges;
	std::map<std::string, bool> created_edges;

	int p1 = lower(points);
	int p2 = next_point(points, p1, -1);

	add_edge(created_edges, open_edges, p2, p1);

	while (!open_edges.empty())
	{
		p1 = open_edges.top()[0];
		p2 = open_edges.top()[1];
		open_edges.pop();

		if (edge_exists(created_edges, p1, p2))
			continue;

		int p3 = next_point(points, p1, p2);

		polys.push_back(p1);
		polys.push_back(p2);
		polys.push_back(p3);

		add_edge(created_edges, open_edges, p1, p2);
		add_edge(created_edges, open_edges, p2, p3);
		add_edge(created_edges, open_edges, p3, p1);
	}
}

int lower(std::vector<vec3> points)
{
	int low = 0;

	for (int i = 1; i < points.size(); i++)
	{
		if (points[i].z < points[low].z)
		{
			low = i;
		}
		else if (fabs(points[i].z - points[low].z) < 0.002)
		{
			if (points[i].y < points[low].y)
			{
				low = i;
			}
			else if (fabs(points[i].y - points[low].y) < 0.002)
			{
				if (points[i].x < points[low].x)
					low = i;
			}
		}
	}

	return low;
}

int next_point(std::vector<vec3> points, int p1_i, int p2_i)
{
	vec3 p1 = points[p1_i];
	vec3 p2;

	if (p2_i < 0)
	{
		vec3 v = vec3(1, 1, 0);
		p2 = p1.sub(v);
	}
	else
	{
		p2 = points[p2_i];
	}

	vec3 edge = p2.sub(p1);
	edge.normalize();

	int candidate_i = -1;

	for (int i = 0; i < points.size(); i++)
	{
		if (i == p1_i || i == p2_i)
			continue;

		if (candidate_i == -1)
		{
			candidate_i = i;
			continue;
		}

		vec3 v = points[i].sub(p1);
		vec3 po = v.project_over(edge);
		v = v.sub(po);

		vec3 candidate = points[candidate_i].sub(p1);
		po = candidate.project_over(edge);
		candidate = candidate.sub(po);

		vec3 cross = candidate.cross(v);
		if (cross.dot(edge) > 0)
			candidate_i = i;
	}

	return candidate_i;
}

//	vec3 tests

void vec3_all_tests()
{
	if (vec3_add_test())
		std::cout << "PASSED" << std::endl << std::endl;
	else
		std::cout << "FAILED" << std::endl << std::endl;

	if (vec3_sub_test())
		std::cout << "PASSED" << std::endl << std::endl;
	else
		std::cout << "FAILED" << std::endl << std::endl;

	if (vec3_cross_test())
		std::cout << "PASSED" << std::endl << std::endl;
	else
		std::cout << "FAILED" << std::endl << std::endl;

	if (vec3_aa_normalize_test())
		std::cout << "PASSED" << std::endl << std::endl;
	else
		std::cout << "FAILED" << std::endl << std::endl;

	if (vec3_length1_normalize_test())
		std::cout << "PASSED" << std::endl << std::endl;
	else
		std::cout << "FAILED" << std::endl << std::endl;

	if (vec3_po_axis_test())
		std::cout << "PASSED" << std::endl << std::endl;
	else
		std::cout << "FAILED" << std::endl << std::endl;

}

bool vec3_add_test()
{
	vec3 a = vec3(1, 2, 3);
	vec3 b = vec3(1, 1, 4);
	vec3 c = a.add(b);

	std::cout << "@" << std::endl;
	std::cout << "add test" << std::endl;
	std::cout << "a is => ";
	a.debug();
	std::cout << "b is => ";
	b.debug();
	std::cout << "c should be a.add(b) => 2 3 7" << std::endl;
	std::cout << "c is => ";
	c.debug();

	return (c.x == 2 && c.y == 3 && c.z == 7);
}

bool vec3_sub_test()
{
	vec3 a = vec3(1, 2, 3);
	vec3 b = vec3(1, 1, 4);
	vec3 c = a.sub(b);

	std::cout << "@" << std::endl;
	std::cout << "sub test" << std::endl;
	std::cout << "a is => ";
	a.debug();
	std::cout << "b is => ";
	b.debug();
	std::cout << "c should be a.sub(b) => 0 1 -1" << std::endl;
	std::cout << "c is => ";
	c.debug();

	return (c.x == 0 && c.y == 1 && c.z == -1);
}

bool vec3_cross_test()
{
	vec3 y = vec3(0, 1, 0);
	vec3 x = vec3(1, 0, 0);
	vec3 v = x.cross(y);

	std::cout << "@" << std::endl;
	std::cout << "cross test" << std::endl;
	std::cout << "y is => ";
	y.debug();
	std::cout << "x is => ";
	x.debug();
	std::cout << "v should be x.cross(y) => 0 0 1" << std::endl;
	std::cout << "v is => ";
	v.debug();

	return (v.x == 0 && v.y == 0 && v.z == 1);
}

bool vec3_aa_normalize_test()
{
	vec3 a = vec3(3, 0, 0);

	std::cout << "@" << std::endl;
	std::cout << "AA normalize test" << std::endl;
	std::cout << "a is => ";
	a.debug();

	a.normalize();

	std::cout << "a should be a.normalize() => 1 0 0" << std::endl;
	std::cout << "a is => ";
	a.debug();

	return (a.x == 1 && a.y == 0 && a.z == 0);
}

bool vec3_length1_normalize_test()
{
	vec3 a = vec3(1, 2, 3);

	std::cout << "@" << std::endl;
	std::cout << "length 1 normalize test" << std::endl;
	std::cout << "a is => ";
	a.debug();

	a.normalize();

	std::cout << "a.x * a.x + a.y * a.y + a.z * a.z (after a.normalize) should be 1" << std::endl;
	std::cout << "and is =>";

	int l = a.x * a.x + a.y * a.y + a.z * a.z;
	float lf = a.x * a.x + a.y * a.y + a.z * a.z;

	std::cout.precision(150);
	std::cout << l << std::endl;
	std::cout << lf << std::endl;

	return (l == 1);
}

bool vec3_po_axis_test()
{
	vec3 a = vec3(1, 1, 0);
	vec3 x = vec3(1, 0, 0);
	vec3 b = a.project_over(x);

	std::cout << "@" << std::endl;
	std::cout << "project over test" << std::endl;
	std::cout << "a is => ";
	a.debug();
	std::cout << "x is => ";
	x.debug();
	std::cout << "b should be a.project_over(x) => 1 0 0" << std::endl;
	std::cout << "b is => ";
	b.debug();

	return (b.x == 1 && b.y == 0 && b.z == 0);
}

void ch_all_tests()
{
	ch_simple_lower_test();
	ch_lower_with_tie_test();
	ch_lower_point_bug_case_test();
	ch_tetrahedron_test();
}

void ch_simple_lower_test()
{
	std::vector<vec3> points;
	points.push_back(vec3(0, 0, 0));
	points.push_back(vec3(0, 0, 1));
	points.push_back(vec3(0, 2, 1));
	sanitize(points);

	std::cout << "@ simple lower test" << std::endl;

	if (lower(points) == 0)
		std::cout << "PASSED" << std::endl << std::endl;
	else
		std::cout << "FAILED" << std::endl << std::endl;
}

void ch_lower_with_tie_test()
{
	std::vector<vec3> points;
	points.push_back(vec3(0, 1, 0));
	points.push_back(vec3(0, 0, 0));
	points.push_back(vec3(0, 1, 1));

	std::cout << "@ lower with tie test" << std::endl;

	if (lower(points) == 1)
		std::cout << "PASSED" << std::endl << std::endl;
	else
		std::cout << "FAILED" << std::endl << std::endl;
}

void ch_lower_point_bug_case_test()
{
	std::vector<vec3> points;
	points.push_back(vec3(-0.58, -0.63, 0.94));
	points.push_back(vec3(-0.56, 0.68, 0.87));
	points.push_back(vec3(-0.11, -0.15, 0.09));
	points.push_back(vec3(-0.53, -0.16, 0.25));
	points.push_back(vec3(0.28, 0.59, 1.13));
	points.push_back(vec3(0.55, 0.89, 0.31));
	points.push_back(vec3(0.30, -0.53, 1.38));
	points.push_back(vec3(0.72, -0.84, 0.95));
	points.push_back(vec3(-0.87, -0.44, 0.94));
	points.push_back(vec3(0.28, -0.87, 0.75));

	std::cout << "@ lower point bug case test" << std::endl;

	if (lower(points) == 2)
		std::cout << "PASSED" << std::endl << std::endl;
	else
		std::cout << "FAILED" << std::endl << std::endl;
}

void ch_tetrahedron_test()
{
	std::vector<vec3> points;
	std::vector<int> polys;

	points.push_back(vec3(0, 0, 0));
	points.push_back(vec3(1, 0, 0));
	points.push_back(vec3(0, 1, 0));
	points.push_back(vec3(0, 0, 1));

	sanitize(points);
	giftwrap(points, polys);

	std::cout << "polys size " << polys.size() / 3 << std::endl;
	for (int i = 0; i < polys.size() / 3; i++)
	{
		std::cout << polys[i*3] << " # " << polys[i*3+1] << " # " << polys[i*3+2] << std::endl;
	}
}
