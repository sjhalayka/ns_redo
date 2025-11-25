/*
 * 2D Navier-Stokes Fluid Simulation using OpenGL 4 Fragment Shaders
 * Features:
 *   - Semi-Lagrangian advection
 *   - Jacobi pressure solver
 *   - Vorticity confinement
 *   - Obstacle texture support
 *   - Mouse-based velocity and density sources
 *   - Proper rectangular (non-square) texture handling
 *
 * Libraries: GLEW, GLM, GLUT (freeglut)
 *
 * Build:
 *   g++ -o fluid main.cpp -lGL -lGLEW -lglut -lGLU -std=c++11
 *
 * Controls:
 *   Left mouse button: Add density
 *   Right mouse button: Add velocity
 *   Middle mouse button / Shift+Left: Add obstacles
 *   'r': Reset simulation
 *   'o': Clear obstacles
 *   'v': Toggle vorticity confinement
 *   '+'/'-': Adjust vorticity strength
 *   'q'/ESC: Quit
 */

/*
implement a C++ 2D Navier-Stokes simulation using OpenGL 4 fragment shaders. use GLEW, GLM, and GLUT libraries. include vorticity confinement. include obstacle texture. include adding mouse-based velocity and density sources. make sure you take into account rectangular (non-square) texture sizes. use one source file main.cpp, no separate shader files
*/





#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>

 // Simulation parameters
const int SIM_WIDTH = 512;
const int SIM_HEIGHT = 384;  // Non-square to demonstrate rectangular handling
const int JACOBI_ITERATIONS = 40;
const float TIME_STEP = 0.016f;
const float DENSITY_DISSIPATION = 0.995f;
const float VELOCITY_DISSIPATION = 0.99f;
const float VORTICITY_SCALE = 0.35f;

// Window dimensions
int windowWidth = 1024;
int windowHeight = 768;

// Mouse state
int mouseX = 0, mouseY = 0;
int lastMouseX = 0, lastMouseY = 0;
bool leftMouseDown = false;
bool rightMouseDown = false;
bool middleMouseDown = false;
bool shiftDown = false;

// Simulation state
bool vorticityEnabled = true;
float vorticityStrength = VORTICITY_SCALE;

// OpenGL objects
GLuint velocityFBO[2], pressureFBO[2], densityFBO[2];
GLuint velocityTex[2], pressureTex[2], densityTex[2];
GLuint divergenceFBO, divergenceTex;
GLuint vorticityFBO_obj, vorticityTex;
GLuint obstacleFBO, obstacleTex;
GLuint tempFBO, tempTex;

int currentVelocity = 0;
int currentPressure = 0;
int currentDensity = 0;

// Shader programs
GLuint advectProgram;
GLuint divergenceProgram;
GLuint jacobiProgram;
GLuint gradientSubtractProgram;
GLuint addSourceProgram;
GLuint boundaryProgram;
GLuint displayProgram;
GLuint vorticityProgram;
GLuint vorticityForceProgram;
GLuint obstacleProgram;
GLuint copyProgram;

// Quad VAO
GLuint quadVAO, quadVBO;

// Shader source code embedded as strings
const char* vertexShaderSource = R"(
#version 400 core
layout(location = 0) in vec2 position;
out vec2 texCoord;
void main() {
    texCoord = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
)";

const char* advectFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D velocity;
uniform sampler2D quantity;
uniform sampler2D obstacles;
uniform vec2 texelSize;
uniform float dt;
uniform float dissipation;

void main() {
    if (texture(obstacles, texCoord).x > 0.5) {
        fragColor = vec4(0.0);
        return;
    }
    
    vec2 vel = texture(velocity, texCoord).xy;
    vec2 pos = texCoord - vel * texelSize * dt;
    
    // Clamp to valid texture coordinates
    pos = clamp(pos, texelSize * 0.5, 1.0 - texelSize * 0.5);
    
    fragColor = dissipation * texture(quantity, pos);
}
)";

const char* divergenceFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D velocity;
uniform sampler2D obstacles;
uniform vec2 texelSize;

void main() {
    float oC = texture(obstacles, texCoord).x;
    if (oC > 0.5) {
        fragColor = vec4(0.0);
        return;
    }
    
    vec2 vL = texture(velocity, texCoord - vec2(texelSize.x, 0.0)).xy;
    vec2 vR = texture(velocity, texCoord + vec2(texelSize.x, 0.0)).xy;
    vec2 vB = texture(velocity, texCoord - vec2(0.0, texelSize.y)).xy;
    vec2 vT = texture(velocity, texCoord + vec2(0.0, texelSize.y)).xy;
    
    float oL = texture(obstacles, texCoord - vec2(texelSize.x, 0.0)).x;
    float oR = texture(obstacles, texCoord + vec2(texelSize.x, 0.0)).x;
    float oB = texture(obstacles, texCoord - vec2(0.0, texelSize.y)).x;
    float oT = texture(obstacles, texCoord + vec2(0.0, texelSize.y)).x;
    
    // Use center velocity for obstacle boundaries (no-slip)
    vec2 vC = texture(velocity, texCoord).xy;
    if (oL > 0.5) vL = -vC;
    if (oR > 0.5) vR = -vC;
    if (oB > 0.5) vB = -vC;
    if (oT > 0.5) vT = -vC;
    
    float divergence = 0.5 * ((vR.x - vL.x) + (vT.y - vB.y));
    fragColor = vec4(divergence, 0.0, 0.0, 1.0);
}
)";

const char* jacobiFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D pressure;
uniform sampler2D divergence;
uniform sampler2D obstacles;
uniform vec2 texelSize;
uniform float alpha;
uniform float rBeta;

void main() {
    float oC = texture(obstacles, texCoord).x;
    if (oC > 0.5) {
        fragColor = vec4(0.0);
        return;
    }
    
    float pL = texture(pressure, texCoord - vec2(texelSize.x, 0.0)).x;
    float pR = texture(pressure, texCoord + vec2(texelSize.x, 0.0)).x;
    float pB = texture(pressure, texCoord - vec2(0.0, texelSize.y)).x;
    float pT = texture(pressure, texCoord + vec2(0.0, texelSize.y)).x;
    float pC = texture(pressure, texCoord).x;
    
    float oL = texture(obstacles, texCoord - vec2(texelSize.x, 0.0)).x;
    float oR = texture(obstacles, texCoord + vec2(texelSize.x, 0.0)).x;
    float oB = texture(obstacles, texCoord - vec2(0.0, texelSize.y)).x;
    float oT = texture(obstacles, texCoord + vec2(0.0, texelSize.y)).x;
    
    // Use center pressure for obstacle boundaries
    if (oL > 0.5) pL = pC;
    if (oR > 0.5) pR = pC;
    if (oB > 0.5) pB = pC;
    if (oT > 0.5) pT = pC;
    
    float bC = texture(divergence, texCoord).x;
    fragColor = vec4((pL + pR + pB + pT + alpha * bC) * rBeta, 0.0, 0.0, 1.0);
}
)";

const char* gradientSubtractFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D pressure;
uniform sampler2D velocity;
uniform sampler2D obstacles;
uniform vec2 texelSize;

void main() {
    float oC = texture(obstacles, texCoord).x;
    if (oC > 0.5) {
        fragColor = vec4(0.0);
        return;
    }
    
    float pL = texture(pressure, texCoord - vec2(texelSize.x, 0.0)).x;
    float pR = texture(pressure, texCoord + vec2(texelSize.x, 0.0)).x;
    float pB = texture(pressure, texCoord - vec2(0.0, texelSize.y)).x;
    float pT = texture(pressure, texCoord + vec2(0.0, texelSize.y)).x;
    
    float oL = texture(obstacles, texCoord - vec2(texelSize.x, 0.0)).x;
    float oR = texture(obstacles, texCoord + vec2(texelSize.x, 0.0)).x;
    float oB = texture(obstacles, texCoord - vec2(0.0, texelSize.y)).x;
    float oT = texture(obstacles, texCoord + vec2(0.0, texelSize.y)).x;
    
    float pC = texture(pressure, texCoord).x;
    
    // Handle obstacle boundaries
    vec2 obstVel = vec2(0.0);
    vec2 vMask = vec2(1.0);
    
    if (oL > 0.5) { pL = pC; vMask.x = 0.0; }
    if (oR > 0.5) { pR = pC; vMask.x = 0.0; }
    if (oB > 0.5) { pB = pC; vMask.y = 0.0; }
    if (oT > 0.5) { pT = pC; vMask.y = 0.0; }
    
    vec2 vel = texture(velocity, texCoord).xy;
    vec2 grad = vec2(pR - pL, pT - pB) * 0.5;
    vec2 newVel = vel - grad;
    
    fragColor = vec4(vMask * newVel + obstVel, 0.0, 1.0);
}
)";

const char* addSourceFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D field;
uniform sampler2D obstacles;
uniform vec2 point;
uniform vec3 value;
uniform float radius;
uniform vec2 texelSize;
uniform vec2 aspectRatio;

void main() {
    if (texture(obstacles, texCoord).x > 0.5) {
        fragColor = texture(field, texCoord);
        return;
    }
    
    vec2 diff = (texCoord - point) * aspectRatio;
    float dist = length(diff);
    float factor = exp(-dist * dist / radius);
    
    vec4 current = texture(field, texCoord);
    fragColor = current + factor * vec4(value, 0.0);
}
)";

const char* vorticityFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D velocity;
uniform sampler2D obstacles;
uniform vec2 texelSize;

void main() {
    if (texture(obstacles, texCoord).x > 0.5) {
        fragColor = vec4(0.0);
        return;
    }
    
    float vL = texture(velocity, texCoord - vec2(texelSize.x, 0.0)).y;
    float vR = texture(velocity, texCoord + vec2(texelSize.x, 0.0)).y;
    float vB = texture(velocity, texCoord - vec2(0.0, texelSize.y)).x;
    float vT = texture(velocity, texCoord + vec2(0.0, texelSize.y)).x;
    
    float vorticity = ((vR - vL) - (vT - vB)) * 0.5;
    fragColor = vec4(vorticity, 0.0, 0.0, 1.0);
}
)";

const char* vorticityForceFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D velocity;
uniform sampler2D vorticity;
uniform sampler2D obstacles;
uniform vec2 texelSize;
uniform float dt;
uniform float scale;

void main() {
    if (texture(obstacles, texCoord).x > 0.5) {
        fragColor = vec4(0.0);
        return;
    }
    
    float vL = texture(vorticity, texCoord - vec2(texelSize.x, 0.0)).x;
    float vR = texture(vorticity, texCoord + vec2(texelSize.x, 0.0)).x;
    float vB = texture(vorticity, texCoord - vec2(0.0, texelSize.y)).x;
    float vT = texture(vorticity, texCoord + vec2(0.0, texelSize.y)).x;
    float vC = texture(vorticity, texCoord).x;
    
    vec2 force = vec2(abs(vT) - abs(vB), abs(vR) - abs(vL)) * 0.5;
    float len = length(force);
    if (len > 0.0001) {
        force = force / len;
    }
    force *= scale * vC * vec2(1.0, -1.0);
    
    vec2 vel = texture(velocity, texCoord).xy;
    fragColor = vec4(vel + force * dt, 0.0, 1.0);
}
)";

const char* obstacleFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D obstacles;
uniform vec2 point;
uniform float radius;
uniform vec2 texelSize;
uniform vec2 aspectRatio;
uniform float addOrRemove;

void main() {
    vec2 diff = (texCoord - point) * aspectRatio;
    float dist = length(diff);
    
    float current = texture(obstacles, texCoord).x;
    
    if (dist < radius) {
        fragColor = vec4(addOrRemove, 0.0, 0.0, 1.0);
    } else {
        fragColor = vec4(current, 0.0, 0.0, 1.0);
    }
}
)";

const char* displayFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D density;
uniform sampler2D velocity;
uniform sampler2D obstacles;

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    float obstacle = texture(obstacles, texCoord).x;
    if (obstacle > 0.5) {
        fragColor = vec4(0.3, 0.3, 0.35, 1.0);
        return;
    }
    
    vec3 d = texture(density, texCoord).rgb;
    vec2 v = texture(velocity, texCoord).xy;
    
    // Color based on density with slight velocity influence
    float speed = length(v);
    float hue = 0.6 - speed * 0.002;  // Blue to red based on speed
    float sat = 0.7;
    float val = clamp(d.r * 2.0, 0.0, 1.0);
    
    vec3 color = hsv2rgb(vec3(hue, sat, val));
    
    // Add some base color for visibility
    color = mix(vec3(0.02, 0.02, 0.05), color, val);
    
    fragColor = vec4(color, 1.0);
}
)";

const char* copyFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D source;

void main() {
    fragColor = texture(source, texCoord);
}
)";

const char* boundaryFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D field;
uniform vec2 texelSize;
uniform float scale;

void main() {
    vec2 coord = texCoord;
    float boundary = 0.0;
    vec2 offset = vec2(0.0);
    
    // Left boundary
    if (coord.x < texelSize.x) {
        offset.x = texelSize.x;
        boundary = 1.0;
    }
    // Right boundary
    else if (coord.x > 1.0 - texelSize.x) {
        offset.x = -texelSize.x;
        boundary = 1.0;
    }
    // Bottom boundary
    if (coord.y < texelSize.y) {
        offset.y = texelSize.y;
        boundary = 1.0;
    }
    // Top boundary
    else if (coord.y > 1.0 - texelSize.y) {
        offset.y = -texelSize.y;
        boundary = 1.0;
    }
    
    if (boundary > 0.5) {
        fragColor = scale * texture(field, coord + offset);
    } else {
        fragColor = texture(field, coord);
    }
}
)";

// Utility functions
void checkGLError(const char* operation) {
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << "OpenGL error after " << operation << ": " << error << std::endl;
    }
}

GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader compilation failed:\n" << infoLog << std::endl;
        std::cerr << "Source:\n" << source << std::endl;
    }

    return shader;
}

GLuint createProgram(const char* vertexSource, const char* fragmentSource) {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSource);

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Program linking failed:\n" << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
}

void createTexture(GLuint& tex, int width, int height, GLenum internalFormat, GLenum format, GLenum type) {
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

void createFBO(GLuint& fbo, GLuint tex) {
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Framebuffer not complete!" << std::endl;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void initShaders() {
    advectProgram = createProgram(vertexShaderSource, advectFragmentSource);
    divergenceProgram = createProgram(vertexShaderSource, divergenceFragmentSource);
    jacobiProgram = createProgram(vertexShaderSource, jacobiFragmentSource);
    gradientSubtractProgram = createProgram(vertexShaderSource, gradientSubtractFragmentSource);
    addSourceProgram = createProgram(vertexShaderSource, addSourceFragmentSource);
    boundaryProgram = createProgram(vertexShaderSource, boundaryFragmentSource);
    displayProgram = createProgram(vertexShaderSource, displayFragmentSource);
    vorticityProgram = createProgram(vertexShaderSource, vorticityFragmentSource);
    vorticityForceProgram = createProgram(vertexShaderSource, vorticityForceFragmentSource);
    obstacleProgram = createProgram(vertexShaderSource, obstacleFragmentSource);
    copyProgram = createProgram(vertexShaderSource, copyFragmentSource);
}

void initTextures() {
    // Velocity textures (RG for x,y components)
    for (int i = 0; i < 2; i++) {
        createTexture(velocityTex[i], SIM_WIDTH, SIM_HEIGHT, GL_RG32F, GL_RG, GL_FLOAT);
        createFBO(velocityFBO[i], velocityTex[i]);
    }

    // Pressure textures
    for (int i = 0; i < 2; i++) {
        createTexture(pressureTex[i], SIM_WIDTH, SIM_HEIGHT, GL_R32F, GL_RED, GL_FLOAT);
        createFBO(pressureFBO[i], pressureTex[i]);
    }

    // Density textures (RGB for colored smoke)
    for (int i = 0; i < 2; i++) {
        createTexture(densityTex[i], SIM_WIDTH, SIM_HEIGHT, GL_RGBA32F, GL_RGBA, GL_FLOAT);
        createFBO(densityFBO[i], densityTex[i]);
    }

    // Divergence texture
    createTexture(divergenceTex, SIM_WIDTH, SIM_HEIGHT, GL_R32F, GL_RED, GL_FLOAT);
    createFBO(divergenceFBO, divergenceTex);

    // Vorticity texture
    createTexture(vorticityTex, SIM_WIDTH, SIM_HEIGHT, GL_R32F, GL_RED, GL_FLOAT);
    createFBO(vorticityFBO_obj, vorticityTex);

    // Obstacle texture
    createTexture(obstacleTex, SIM_WIDTH, SIM_HEIGHT, GL_R32F, GL_RED, GL_FLOAT);
    createFBO(obstacleFBO, obstacleTex);

    // Temp texture for boundary operations
    createTexture(tempTex, SIM_WIDTH, SIM_HEIGHT, GL_RGBA32F, GL_RGBA, GL_FLOAT);
    createFBO(tempFBO, tempTex);

    // Clear all textures
    glBindFramebuffer(GL_FRAMEBUFFER, obstacleFBO);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    for (int i = 0; i < 2; i++) {
        glBindFramebuffer(GL_FRAMEBUFFER, velocityFBO[i]);
        glClear(GL_COLOR_BUFFER_BIT);
        glBindFramebuffer(GL_FRAMEBUFFER, pressureFBO[i]);
        glClear(GL_COLOR_BUFFER_BIT);
        glBindFramebuffer(GL_FRAMEBUFFER, densityFBO[i]);
        glClear(GL_COLOR_BUFFER_BIT);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void initQuad() {
    float quadVertices[] = {
        -1.0f, -1.0f,
         1.0f, -1.0f,
        -1.0f,  1.0f,
         1.0f,  1.0f
    };

    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);

    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

    glBindVertexArray(0);
}

void drawQuad() {
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
}

void setTextureUniform(GLuint program, const char* name, int unit, GLuint texture) {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D, texture);
    glUniform1i(glGetUniformLocation(program, name), unit);
}

void advect(GLuint velocityTex, GLuint quantityTex, GLuint outputFBO, float dissipation) {
    glBindFramebuffer(GL_FRAMEBUFFER, outputFBO);
    glViewport(0, 0, SIM_WIDTH, SIM_HEIGHT);

    glUseProgram(advectProgram);
    setTextureUniform(advectProgram, "velocity", 0, velocityTex);
    setTextureUniform(advectProgram, "quantity", 1, quantityTex);
    setTextureUniform(advectProgram, "obstacles", 2, obstacleTex);
    glUniform2f(glGetUniformLocation(advectProgram, "texelSize"), 1.0f / SIM_WIDTH, 1.0f / SIM_HEIGHT);
    glUniform1f(glGetUniformLocation(advectProgram, "dt"), TIME_STEP * 100.0f);
    glUniform1f(glGetUniformLocation(advectProgram, "dissipation"), dissipation);

    drawQuad();
}

void computeDivergence() {
    glBindFramebuffer(GL_FRAMEBUFFER, divergenceFBO);
    glViewport(0, 0, SIM_WIDTH, SIM_HEIGHT);

    glUseProgram(divergenceProgram);
    setTextureUniform(divergenceProgram, "velocity", 0, velocityTex[currentVelocity]);
    setTextureUniform(divergenceProgram, "obstacles", 1, obstacleTex);
    glUniform2f(glGetUniformLocation(divergenceProgram, "texelSize"), 1.0f / SIM_WIDTH, 1.0f / SIM_HEIGHT);

    drawQuad();
}

void clearPressure() {
    glBindFramebuffer(GL_FRAMEBUFFER, pressureFBO[currentPressure]);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}

void jacobi() {
    glUseProgram(jacobiProgram);
    glUniform2f(glGetUniformLocation(jacobiProgram, "texelSize"), 1.0f / SIM_WIDTH, 1.0f / SIM_HEIGHT);
    glUniform1f(glGetUniformLocation(jacobiProgram, "alpha"), -1.0f);
    glUniform1f(glGetUniformLocation(jacobiProgram, "rBeta"), 0.25f);

    for (int i = 0; i < JACOBI_ITERATIONS; i++) {
        int dst = 1 - currentPressure;
        glBindFramebuffer(GL_FRAMEBUFFER, pressureFBO[dst]);
        glViewport(0, 0, SIM_WIDTH, SIM_HEIGHT);

        setTextureUniform(jacobiProgram, "pressure", 0, pressureTex[currentPressure]);
        setTextureUniform(jacobiProgram, "divergence", 1, divergenceTex);
        setTextureUniform(jacobiProgram, "obstacles", 2, obstacleTex);

        drawQuad();
        currentPressure = dst;
    }
}

void subtractGradient() {
    int dst = 1 - currentVelocity;
    glBindFramebuffer(GL_FRAMEBUFFER, velocityFBO[dst]);
    glViewport(0, 0, SIM_WIDTH, SIM_HEIGHT);

    glUseProgram(gradientSubtractProgram);
    setTextureUniform(gradientSubtractProgram, "pressure", 0, pressureTex[currentPressure]);
    setTextureUniform(gradientSubtractProgram, "velocity", 1, velocityTex[currentVelocity]);
    setTextureUniform(gradientSubtractProgram, "obstacles", 2, obstacleTex);
    glUniform2f(glGetUniformLocation(gradientSubtractProgram, "texelSize"), 1.0f / SIM_WIDTH, 1.0f / SIM_HEIGHT);

    drawQuad();
    currentVelocity = dst;
}

void computeVorticity() {
    glBindFramebuffer(GL_FRAMEBUFFER, vorticityFBO_obj);
    glViewport(0, 0, SIM_WIDTH, SIM_HEIGHT);

    glUseProgram(vorticityProgram);
    setTextureUniform(vorticityProgram, "velocity", 0, velocityTex[currentVelocity]);
    setTextureUniform(vorticityProgram, "obstacles", 1, obstacleTex);
    glUniform2f(glGetUniformLocation(vorticityProgram, "texelSize"), 1.0f / SIM_WIDTH, 1.0f / SIM_HEIGHT);

    drawQuad();
}

void applyVorticityForce() {
    int dst = 1 - currentVelocity;
    glBindFramebuffer(GL_FRAMEBUFFER, velocityFBO[dst]);
    glViewport(0, 0, SIM_WIDTH, SIM_HEIGHT);

    glUseProgram(vorticityForceProgram);
    setTextureUniform(vorticityForceProgram, "velocity", 0, velocityTex[currentVelocity]);
    setTextureUniform(vorticityForceProgram, "vorticity", 1, vorticityTex);
    setTextureUniform(vorticityForceProgram, "obstacles", 2, obstacleTex);
    glUniform2f(glGetUniformLocation(vorticityForceProgram, "texelSize"), 1.0f / SIM_WIDTH, 1.0f / SIM_HEIGHT);
    glUniform1f(glGetUniformLocation(vorticityForceProgram, "dt"), TIME_STEP);
    glUniform1f(glGetUniformLocation(vorticityForceProgram, "scale"), vorticityStrength);

    drawQuad();
    currentVelocity = dst;
}

void addSource(GLuint* textures, GLuint* fbos, int& current, float x, float y, float vx, float vy, float vz, float radius) {
    int dst = 1 - current;
    glBindFramebuffer(GL_FRAMEBUFFER, fbos[dst]);
    glViewport(0, 0, SIM_WIDTH, SIM_HEIGHT);

    glUseProgram(addSourceProgram);
    setTextureUniform(addSourceProgram, "field", 0, textures[current]);
    setTextureUniform(addSourceProgram, "obstacles", 1, obstacleTex);
    glUniform2f(glGetUniformLocation(addSourceProgram, "point"), x, y);
    glUniform3f(glGetUniformLocation(addSourceProgram, "value"), vx, vy, vz);
    glUniform1f(glGetUniformLocation(addSourceProgram, "radius"), radius);
    glUniform2f(glGetUniformLocation(addSourceProgram, "texelSize"), 1.0f / SIM_WIDTH, 1.0f / SIM_HEIGHT);
    // Account for aspect ratio in splat
    glUniform2f(glGetUniformLocation(addSourceProgram, "aspectRatio"), (float)SIM_WIDTH / SIM_HEIGHT, 1.0f);

    drawQuad();
    current = dst;
}

void addObstacle(float x, float y, float radius, bool add) {
    glBindFramebuffer(GL_FRAMEBUFFER, tempFBO);
    glViewport(0, 0, SIM_WIDTH, SIM_HEIGHT);

    glUseProgram(obstacleProgram);
    setTextureUniform(obstacleProgram, "obstacles", 0, obstacleTex);
    glUniform2f(glGetUniformLocation(obstacleProgram, "point"), x, y);
    glUniform1f(glGetUniformLocation(obstacleProgram, "radius"), radius);
    glUniform2f(glGetUniformLocation(obstacleProgram, "texelSize"), 1.0f / SIM_WIDTH, 1.0f / SIM_HEIGHT);
    glUniform2f(glGetUniformLocation(obstacleProgram, "aspectRatio"), (float)SIM_WIDTH / SIM_HEIGHT, 1.0f);
    glUniform1f(glGetUniformLocation(obstacleProgram, "addOrRemove"), add ? 1.0f : 0.0f);

    drawQuad();

    // Copy back to obstacle texture
    glBindFramebuffer(GL_FRAMEBUFFER, obstacleFBO);
    glUseProgram(copyProgram);
    setTextureUniform(copyProgram, "source", 0, tempTex);
    drawQuad();
}

void simulate() {
    // Advect velocity
    advect(velocityTex[currentVelocity], velocityTex[currentVelocity], velocityFBO[1 - currentVelocity], VELOCITY_DISSIPATION);
    currentVelocity = 1 - currentVelocity;

    // Advect density
    advect(velocityTex[currentVelocity], densityTex[currentDensity], densityFBO[1 - currentDensity], DENSITY_DISSIPATION);
    currentDensity = 1 - currentDensity;

    // Vorticity confinement
    if (vorticityEnabled) {
        computeVorticity();
        applyVorticityForce();
    }

    // Pressure projection
    computeDivergence();
    clearPressure();
    jacobi();
    subtractGradient();
}

void display() {
    // Run simulation step
    simulate();

    // Add continuous sources based on mouse input
    if (leftMouseDown && !shiftDown) {
        float x = (float)mouseX / windowWidth;
        float y = 1.0f - (float)mouseY / windowHeight;

        // Add density with varying color based on position
        float hue = fmod(glutGet(GLUT_ELAPSED_TIME) * 0.001f, 1.0f);
        float r = 0.5f + 0.5f * sin(hue * 6.28318f);
        float g = 0.5f + 0.5f * sin(hue * 6.28318f + 2.094f);
        float b = 0.5f + 0.5f * sin(hue * 6.28318f + 4.189f);
        addSource(densityTex, densityFBO, currentDensity, x, y, r * 0.8f, g * 0.8f, b * 0.8f, 0.0008f);
    }

    if (rightMouseDown) {
        float x = (float)mouseX / windowWidth;
        float y = 1.0f - (float)mouseY / windowHeight;
        float dx = (float)(mouseX - lastMouseX) * 2.0f;
        float dy = (float)(lastMouseY - mouseY) * 2.0f;

        addSource(velocityTex, velocityFBO, currentVelocity, x, y, dx, dy, 0.0f, 0.0004f);
    }

    if (middleMouseDown || (leftMouseDown && shiftDown)) {
        float x = (float)mouseX / windowWidth;
        float y = 1.0f - (float)mouseY / windowHeight;
        addObstacle(x, y, 0.02f, true);
    }

    lastMouseX = mouseX;
    lastMouseY = mouseY;

    // Render to screen
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, windowWidth, windowHeight);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(displayProgram);
    setTextureUniform(displayProgram, "density", 0, densityTex[currentDensity]);
    setTextureUniform(displayProgram, "velocity", 1, velocityTex[currentVelocity]);
    setTextureUniform(displayProgram, "obstacles", 2, obstacleTex);

    drawQuad();

    glutSwapBuffers();
    glutPostRedisplay();
}

void reshape(int w, int h) {
    windowWidth = w;
    windowHeight = h;
}

void keyboard(unsigned char key, int x, int y) {
    switch (key) {
    case 27:  // ESC
    case 'q':
    case 'Q':
        exit(0);
        break;
    case 'r':
    case 'R':
        // Reset simulation
        for (int i = 0; i < 2; i++) {
            glBindFramebuffer(GL_FRAMEBUFFER, velocityFBO[i]);
            glClear(GL_COLOR_BUFFER_BIT);
            glBindFramebuffer(GL_FRAMEBUFFER, pressureFBO[i]);
            glClear(GL_COLOR_BUFFER_BIT);
            glBindFramebuffer(GL_FRAMEBUFFER, densityFBO[i]);
            glClear(GL_COLOR_BUFFER_BIT);
        }
        std::cout << "Simulation reset" << std::endl;
        break;
    case 'o':
    case 'O':
        // Clear obstacles
        glBindFramebuffer(GL_FRAMEBUFFER, obstacleFBO);
        glClear(GL_COLOR_BUFFER_BIT);
        std::cout << "Obstacles cleared" << std::endl;
        break;
    case 'v':
    case 'V':
        vorticityEnabled = !vorticityEnabled;
        std::cout << "Vorticity confinement: " << (vorticityEnabled ? "ON" : "OFF") << std::endl;
        break;
    case '+':
    case '=':
        vorticityStrength += 0.05f;
        std::cout << "Vorticity strength: " << vorticityStrength << std::endl;
        break;
    case '-':
    case '_':
        vorticityStrength = std::max(0.0f, vorticityStrength - 0.05f);
        std::cout << "Vorticity strength: " << vorticityStrength << std::endl;
        break;
    }
}

void specialKeys(int key, int x, int y) {
    if (glutGetModifiers() & GLUT_ACTIVE_SHIFT) {
        shiftDown = true;
    }
}

void specialKeysUp(int key, int x, int y) {
    shiftDown = false;
}

void mouse(int button, int state, int x, int y) {
    mouseX = x;
    mouseY = y;

    shiftDown = (glutGetModifiers() & GLUT_ACTIVE_SHIFT) != 0;

    if (button == GLUT_LEFT_BUTTON) {
        leftMouseDown = (state == GLUT_DOWN);
    }
    else if (button == GLUT_RIGHT_BUTTON) {
        rightMouseDown = (state == GLUT_DOWN);
    }
    else if (button == GLUT_MIDDLE_BUTTON) {
        middleMouseDown = (state == GLUT_DOWN);
    }
}

void motion(int x, int y) {
    mouseX = x;
    mouseY = y;
}

void passiveMotion(int x, int y) {
    lastMouseX = mouseX;
    lastMouseY = mouseY;
    mouseX = x;
    mouseY = y;
}

void printControls() {
    std::cout << "\n=== 2D Navier-Stokes Fluid Simulation ===" << std::endl;
    std::cout << "Simulation size: " << SIM_WIDTH << "x" << SIM_HEIGHT << " (non-square)" << std::endl;
    std::cout << "\nControls:" << std::endl;
    std::cout << "  Left mouse button:    Add density (colored smoke)" << std::endl;
    std::cout << "  Right mouse button:   Add velocity" << std::endl;
    std::cout << "  Middle mouse / Shift+Left: Add obstacles" << std::endl;
    std::cout << "  'r': Reset simulation" << std::endl;
    std::cout << "  'o': Clear obstacles" << std::endl;
    std::cout << "  'v': Toggle vorticity confinement" << std::endl;
    std::cout << "  '+'/'-': Adjust vorticity strength" << std::endl;
    std::cout << "  'q'/ESC: Quit" << std::endl;
    std::cout << "\nVorticity confinement: ON" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

#pragma comment(lib, "freeglut")
#pragma comment(lib, "glew32")

int main(int argc, char** argv) {
    // Initialize GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(windowWidth, windowHeight);
    glutInitContextVersion(4, 0);
    glutInitContextProfile(GLUT_CORE_PROFILE);
    glutCreateWindow("2D Navier-Stokes Fluid Simulation");

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "GLEW initialization failed: " << glewGetErrorString(err) << std::endl;
        return -1;
    }

    // Clear any errors from GLEW init
    while (glGetError() != GL_NO_ERROR);

    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;

    // Initialize
    initShaders();
    initTextures();
    initQuad();

    printControls();

    // Register callbacks
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(specialKeys);
    glutSpecialUpFunc(specialKeysUp);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutPassiveMotionFunc(passiveMotion);

    // Enable blending for nice visuals
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    // Main loop
    glutMainLoop();

    return 0;
}