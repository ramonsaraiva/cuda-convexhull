#ifndef INPUT_H
#define INPUT_H	1

#include <SDL/SDL.h>

class InputController
{
	public:
		InputController();
		void events();
		void keyboard_down(SDL_KeyboardEvent event);
		void keyboard_up(SDL_KeyboardEvent event);
		void mouse_down(SDL_MouseButtonEvent event);
		void mouse_up(SDL_MouseButtonEvent event);
		void mousemotion(SDL_MouseMotionEvent event);

	private:
		int _moving_ws;
		int _moving_da;
		int _direction_ws;
		int _direction_da;
		int _looking;
};

#endif
