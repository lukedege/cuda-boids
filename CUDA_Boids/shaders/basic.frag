// Fragment shader
// The output of a fragment shader is the color of the generated fragments
// #version 410 core

// output variable for the fragment shader. Usually, it is the final color of the fragment
out vec4 color;

void main()
{
    color = vec4(1.0f, 1.0f, 0.0f, 1.0f);
}
