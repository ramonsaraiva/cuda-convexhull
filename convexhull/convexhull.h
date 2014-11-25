#ifndef CONVEXHULL
#define CONVEXHULL	1

#include <math.h>
#include <vector>
#include <map>
#include <stack>
#include <array>

#include <iostream>

#include "vec/vec.h"

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
