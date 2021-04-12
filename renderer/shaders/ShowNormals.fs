#version 330 core
out vec4 FragColor;

in vec3 Normal;  
in vec3 FragPos;  
  
void main()
{
    vec3 norm = normalize(Normal);
    vec3 result = norm * 0.5 + 0.5;
    FragColor = vec4(result, 1.0);
}
