#ifndef SCENE_H
#define SCENE_H	1

#include <string>
#include <vector>
#include <map>

#include <GL/glew.h>

#include "tinyobjloader/tiny_obj_loader.h"

#include "camera/camera.h"
#include "convexhull/convexhull.h"

//	SceneObject

class SceneObject
{
	public:
		SceneObject(std::string ident);
		void set_translate(const float x, const float y, const float z);
		void set_rotate(const float x, const float y, const float z);
		void set_scale(const float x, const float y, const float z);
		void set_angle(const float a);
		void set_render_mode(const GLuint mode);
		void set_p(float x, float y, float z);
		void load_obj(const char* file);
		void build_vbo();
		void render();

	public:
		std::string ident();
		std::vector<tinyobj::shape_t>& shapes();
		std::vector<tinyobj::material_t>& materials();
		float* translate();
		float* rotate();
		float* scale();

		void points(std::vector<vec3>& p);
		GLuint render_mode();
		std::string texname();
	
	private:
		std::string _ident;
		std::vector<tinyobj::shape_t> _shapes;
		std::vector<tinyobj::material_t> _materials;
		GLuint _pos_vboid;
		GLuint _normal_vboid;
		GLuint _uv_vboid;
		GLuint _idx_vboid;
		int _idx_size;
		GLuint _render_mode;
		float _p[3];

		std::string _matdir;
		std::string _texname;

		float _translate[3];
		float _rotate[3];
		float _scale[3];
		float _angle;
};

//	Scene

class Scene
{
	public:
		static Scene& instance()
		{
			static Scene instance;
			return instance;
		}

		void add_camera(std::string ident, Camera* camera);
		void set_default_camera(std::string ident);
		void set_default_camera(Camera* cam);

		void add_object(std::string ident, SceneObject* object);

	public:
		void render();
		Camera* default_camera();
		std::map<std::string, SceneObject*>& objects();
		std::map<std::string, GLuint>& textures();

		void dump();

	private:
		Scene() {}
		Scene(Scene const&);
		void operator=(Scene const&);

		std::map<std::string, Camera*> _cameras;
		Camera* _default_camera;

		std::map<std::string, SceneObject*> _objects;
		std::map<std::string, GLuint> _textures;
};

#endif
