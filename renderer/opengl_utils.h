#pragma once

#include "commons.h"

#include <GL/glew.h>
#include <iostream>

BEGIN_RENDERER_NAMESPACE

inline void checkOpenGLError()
{
	GLenum err;
	while ((err = glGetError()) != GL_NO_ERROR)
	{
		std::cerr << "OpenGL Error: " << gluErrorString(err) << std::endl;
		//__debugbreak();
	}
}

END_RENDERER_NAMESPACE
