// Vertex shader
// The output of a vertex shader is the position of the vertex
// after any kind of elaboration
// #version 430 core

layout (location = 0) in vec3 position;

out gl_PerVertex { vec4 gl_Position; };

void main()
{
   gl_Position = vec4(position.x, position.y, position.z, 1.0f);
}