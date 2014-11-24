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

