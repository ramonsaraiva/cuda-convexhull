#include <string.h>

#include <iostream>
#include <string>
#include <vector>
#include <map>

#include <GL/glew.h>
#include <SOIL/SOIL.h>

#include "tinyobjloader/tiny_obj_loader.h"

#include "scene.h"
#include "vec/vec.h"

//	SceneObject

SceneObject::SceneObject(std::string ident)
{
	_ident = ident;

	memset(_translate, 0, 3 * sizeof(float));
	memset(_rotate, 0, 3 * sizeof(float));

	_scale[0] = _scale[1] = _scale[2] = 1.0f;
	_angle = 0;
}

void SceneObject::set_translate(const float x, const float y, const float z)
{
	_translate[0] = x;
	_translate[1] = y;
	_translate[2] = z;
}

void SceneObject::set_rotate(const float x, const float y, const float z)
{
	_rotate[0] = x;
	_rotate[1] = y;
	_rotate[2] = z;
}

void SceneObject::set_scale(const float x, const float y, const float z)
{
	_scale[0] = x;
	_scale[1] = y;
	_scale[2] = z;
}

void SceneObject::set_angle(const float a)
{
	_angle = a;
}

void SceneObject::set_render_mode(GLuint mode)
{
	_render_mode = mode;
}

void SceneObject::set_p(float x, float y, float z)
{
	_p[0] = x;
	_p[1] = y;
	_p[2] = z;
}

void SceneObject::load_obj(const char* file)
{
	std::string f = std::string(file);
	unsigned found = f.find_last_of("\\/");

	if (found != std::string::npos)
	{
		f = f.substr(0, found);
		f = f.append("/");

		tinyobj::LoadObj(_shapes, _materials, file, f.c_str());
		_matdir = f;
	}
	else
	{
		tinyobj::LoadObj(_shapes, _materials, file, NULL);
	}
}

void SceneObject::build_vbo()
{
	std::vector<GLfloat> pos;
	std::vector<GLfloat> normal;
	std::vector<GLfloat> uv;
	std::vector<GLuint> idx;

	for (int i = 0; i < _shapes.size(); i++)
	{
		for (int j = 0; j < _shapes[i].mesh.positions.size(); j++)
			pos.push_back(_shapes[i].mesh.positions[j]);

		for (int j = 0; j < _shapes[i].mesh.normals.size(); j++)
			normal.push_back(_shapes[i].mesh.normals[j]);

		for (int j = 0; j < _shapes[i].mesh.texcoords.size(); j++)
			uv.push_back(_shapes[i].mesh.texcoords[j]);

		for (int j = 0; j < _shapes[i].mesh.indices.size(); j++)
			idx.push_back(_shapes[i].mesh.indices[j]);
	}

	glGenBuffers(1, &_pos_vboid);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _pos_vboid);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLfloat) * pos.size(), &pos.front(), GL_STATIC_DRAW);

	_idx_size = idx.size();

	glGenBuffers(1, &_normal_vboid);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _normal_vboid);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLfloat) * normal.size(), &normal.front(), GL_STATIC_DRAW);

	glGenBuffers(1, &_uv_vboid);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _uv_vboid);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLfloat) * uv.size(), &uv.front(), GL_STATIC_DRAW);

	glGenBuffers(1, &_idx_vboid);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _idx_vboid);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLfloat) * idx.size(), &idx.front(), GL_STATIC_DRAW);

	if (_materials.size() > 0)
	{
		_texname = _materials[0].diffuse_texname;

		if (Scene::instance().textures().find(texname()) == Scene::instance().textures().end())
		{
			GLuint texid = SOIL_load_OGL_texture(texname().c_str(), SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, SOIL_FLAG_MIPMAPS | SOIL_FLAG_INVERT_Y | SOIL_FLAG_NTSC_SAFE_RGB | SOIL_FLAG_COMPRESS_TO_DXT);
			Scene::instance().textures()[texname()] = texid;
		}
	}
}

void SceneObject::render()
{
	if (_render_mode == GL_POINTS)
		glColor3f(1, 1, 1);

	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	glEnableClientState(GL_VERTEX_ARRAY);

	glBindBuffer(GL_ARRAY_BUFFER, _pos_vboid);
	glVertexPointer(3, GL_FLOAT, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, _normal_vboid);
	glNormalPointer(GL_FLOAT, 0, 0);

	if (!_texname.empty())
	{
		glBindTexture(GL_TEXTURE_2D, Scene::instance().textures()[texname()]);

		//std::cout << "loaded tex " << texname() << " with id " << Scene::instance().textures()[texname()] << std::endl;

		glBindBuffer(GL_ARRAY_BUFFER, _uv_vboid);
		glTexCoordPointer(2, GL_FLOAT, 0, 0);
	}

	glPushMatrix();

	glTranslatef(_translate[0], _translate[1], _translate[2]);
	glRotatef(_rotate[0], 1, 0, 0);
	glRotatef(_rotate[1], 0, 1, 0);
	glRotatef(_rotate[2], 0, 0, 1);
	glScalef(_scale[0], _scale[1], _scale[2]);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _idx_vboid);
	glDrawElements(_render_mode, _idx_size, GL_UNSIGNED_INT, NULL);

	glPointSize(20);
	glColor3f(0, 1, 0);
	glBegin(GL_POINTS);
		glVertex3f(_p[0], _p[1], _p[2]);
	glEnd();
	glPointSize(10);

	glPopMatrix();

	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);

	glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
	glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);
}

std::string SceneObject::ident()
{
	return _ident;
}

std::vector<tinyobj::shape_t>& SceneObject::shapes()
{
	return _shapes;
}

std::vector<tinyobj::material_t>& SceneObject::materials()
{
	return _materials;
}

float* SceneObject::translate()
{
	return _translate;
}

float* SceneObject::rotate()
{
	return _rotate;
}

float* SceneObject::scale()
{
	return _scale;
}

void SceneObject::points(std::vector<vec3>& p)
{
	for (int i = 0; i < _shapes.size(); i++)
	{
		for (int j = 0; j < _shapes[i].mesh.positions.size() / 3; j++)
		{
			struct vec3_s v = vec3_s(_shapes[i].mesh.positions[j*3], _shapes[i].mesh.positions[j*3+1], _shapes[i].mesh.positions[j*3+2]);

			p.push_back(v);
		}
	}
}

GLuint SceneObject::render_mode()
{
	return _render_mode;
}

std::string SceneObject::texname()
{
	return _matdir + _texname;
}

//	Scene

void Scene::add_camera(std::string ident, Camera* camera)
{
	_cameras[ident] = camera;
}

void Scene::set_default_camera(std::string ident)
{
	_default_camera = _cameras[ident];
}

void Scene::set_default_camera(Camera* cam)
{
	_default_camera = cam;
}

void Scene::add_object(std::string ident, SceneObject* object)
{
	_objects[ident] = object;
}

void Scene::render()
{
	typedef std::map<std::string, SceneObject*>::iterator it_type;
	for (it_type i = _objects.begin(); i != _objects.end(); i++)
	{
		i->second->render();
	}
}

Camera* Scene::default_camera()
{
	return _default_camera;
}

std::map<std::string, SceneObject*>& Scene::objects()
{
	return _objects;
}

std::map<std::string, GLuint>& Scene::textures()
{
	return _textures;
}

void Scene::dump()
{
	//dump cameras and sceneobjects
}
