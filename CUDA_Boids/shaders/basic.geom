// Geometry shader
// The output of a geometry shader is one or more primitives, taken from the vertex shader and transformed as necessary
// This basic geometry shader act as a mere passthrough, not modifying anything from the vertex shader
// #version 430 core

// input from the vertex shader
layout (points) in;
// output for the geometry shader, can be different primitives (points, lines or triangles)
layout (points, max_vertices = 1) out; 

void main() 
{    
    gl_Position = gl_in[0].gl_Position; 
    EmitVertex();
    EndPrimitive();
}  