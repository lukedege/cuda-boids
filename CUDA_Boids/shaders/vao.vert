// Vertex shader
// #version 430 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 velocity;

out vec3 velocity_norm;

void main()
{
	velocity_norm = normalize(velocity);
	gl_Position = vec4(position, 1.0f);
}