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

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <unordered_map>

 // Simulation parameters
const int SIM_WIDTH = 1920;
const int SIM_HEIGHT = 1080;
const int JACOBI_ITERATIONS = 20;
const float DENSITY_DISSIPATION = 0.975f;
const float VELOCITY_DISSIPATION = 0.99999f;
const float VORTICITY_SCALE = 10.0f;

bool red_mode = true;

float GLOBAL_TIME = 0;
const float FPS = 120;
float DT = 1.0f / FPS;


// Window dimensions
int windowWidth = 1920;
int windowHeight = 1080;

// Mouse state
int mouseX = 0, mouseY = 0;
int lastMouseX = 0, lastMouseY = 0;
bool leftMouseDown = false;
bool rightMouseDown = false;
bool middleMouseDown = false;
bool shiftDown = false;

// Simulation state
//bool vorticityEnabled = true;
float vorticityStrength = VORTICITY_SCALE;

// Protagonist sprite texture
GLuint protagonistTex = 0;
int protagonistWidth = 0;
int protagonistHeight = 0;

GLuint foregroundTex = 0;
int foregroundWidth = 0;
int foregroundHeight = 0;

GLuint backgroundTex = 0;
int backgroundWidth = 0;
int backgroundHeight = 0;





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
GLuint obstacleStampProgram;
GLuint copyProgram;
GLuint spriteProgram;

// Quad VAO
GLuint quadVAO, quadVBO;







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



const char* textVertexShaderSource = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoord;
layout(location = 2) in vec4 aColor;

uniform mat4 projection;
uniform mat4 model;

out vec2 TexCoord;
out vec4 Color;

void main() {
    gl_Position = projection * model * vec4(aPos, 1.0);
    TexCoord = aTexCoord;
    Color = aColor;
}
)";

const char* textFragmentShaderSource = R"(
#version 330 core
in vec2 TexCoord;
in vec4 Color;

uniform sampler2D fontTexture;
uniform bool useColor;

out vec4 FragColor;

void main() {
    vec4 texColor = texture(fontTexture, TexCoord);
    
    // Handle different font texture formats
    if (useColor) {
        // For colored font atlas, just blend with the vertex color
        FragColor = texColor * Color;
    } else {
        // For grayscale/alpha font atlas, use the alpha/red channel
        // with the vertex color
        float alpha = texColor.r; // or texColor.a depending on your font texture
        FragColor = vec4(Color.rgb, Color.a * alpha);
    }
}
)";





struct FontAtlas {
    GLuint textureID;
    int charWidth;      // Width of each character (16)
    int charHeight;     // Height of each character (16)
    int atlasWidth;     // Atlas width (256)
    int atlasHeight;    // Atlas height (256)
    int charsPerRow;    // Characters per row in the atlas (16)
};


GLuint loadFontTexture(const char* filename) {
    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Load image using stb_image (you're already using this)
    int width, height, channels;
    stbi_set_flip_vertically_on_load(false);


    unsigned char* data = stbi_load(filename, &width, &height, &channels, 0);

    if (!data) {
        std::cerr << "Failed to load font texture: " << filename << std::endl;
        std::cerr << "STB Image error: " << stbi_failure_reason() << std::endl;
        return 0;
    }

    // Determine format based on channels
    GLenum format;
    switch (channels) {
    case 1: format = GL_RED; break;
    case 3: format = GL_RGB; break;
    case 4: format = GL_RGBA; break;
    default:
        format = GL_RGB;
        std::cerr << "Unsupported number of channels: " << channels << std::endl;
    }

    // Load texture data to GPU
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);

    // Free image data
    stbi_image_free(data);

    return textureID;
}

FontAtlas initFontAtlas(const char* filename) {
    FontAtlas atlas;
    atlas.textureID = loadFontTexture(filename);
    atlas.charWidth = 64;
    atlas.charHeight = 64;
    atlas.atlasWidth = 1024;
    atlas.atlasHeight = 1024;
    atlas.charsPerRow = atlas.atlasWidth / atlas.charWidth; // 16

    return atlas;
}


class TextRenderer {
private:
    FontAtlas atlas;
    GLuint VAO, VBO, EBO;
    GLuint shaderProgram;
    glm::mat4 projection;

    struct Vertex {
        glm::vec3 position;
        glm::vec2 texCoord;
        glm::vec4 color;
    };

public:

    std::unordered_map<char, int> charWidths; // Map to store actual widths

    // Add this method to calculate character widths
    void calculateCharacterWidths() {
        // Read back the font texture data from GPU
        int dataSize = atlas.atlasWidth * atlas.atlasHeight * 4; // RGBA format
        std::vector<unsigned char> textureData(dataSize);

        glBindTexture(GL_TEXTURE_2D, atlas.textureID);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureData.data());





        // Analyze each character
        for (unsigned short c_ = 0; c_ < 256; c_++)
        {
            unsigned char c = static_cast<unsigned char>(c_);

            // ASCII range
            int atlasX = (c % atlas.charsPerRow) * atlas.charWidth;
            int atlasY = (c / atlas.charsPerRow) * atlas.charHeight;

            // Find leftmost non-empty column
            int leftEdge = atlas.charWidth - 1; // Start from rightmost position

            // Find rightmost non-empty column
            int rightEdge = 0; // Start from leftmost position

            // Scan all columns for this character
            for (int x = 0; x < atlas.charWidth; x++) {
                bool columnHasPixels = false;

                // Check if any pixel in this column is non-transparent
                for (int y = 0; y < atlas.charHeight; y++) {
                    int pixelIndex = ((atlasY + y) * atlas.atlasWidth + (atlasX + x)) * 4;
                    if (pixelIndex >= 0 && pixelIndex < dataSize - 3) {
                        // Check alpha value (using red channel for grayscale font)
                        if (textureData[pixelIndex] > 20) { // Non-transparent threshold
                            columnHasPixels = true;
                            break;
                        }
                    }
                }

                if (columnHasPixels) {
                    // Update left edge (minimum value)
                    leftEdge = std::min(leftEdge, x);
                    // Update right edge (maximum value)
                    rightEdge = std::max(rightEdge, x);
                }
            }

            // If no pixels were found (space or empty character)
            if (rightEdge < leftEdge) {
                // Default width for space character
                if (c == ' ') {
                    charWidths[c] = atlas.charWidth / 3; // Make space 1/3 of cell width
                }
                else {
                    charWidths[c] = atlas.charWidth / 4; // Default minimum width
                }
            }
            else {
                // Calculate width based on the actual character bounds
                int actualWidth = (rightEdge - leftEdge) + 1;

                // Add some padding
                int paddedWidth = actualWidth + 4; // 2 pixels on each side

                // Store this character's width (minimum width of 1/4 of the cell)
                charWidths[c] = std::max(paddedWidth, atlas.charWidth / 4);
            }
        }
    }


    TextRenderer(const char* fontAtlasFile, int windowWidth, int windowHeight) {
        // Initialize font atlas
        atlas = initFontAtlas(fontAtlasFile);

        // Create shader program
        GLuint vertexShader = compileShader(GL_VERTEX_SHADER, textVertexShaderSource);
        GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, textFragmentShaderSource);

        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);

        // Check for linking errors
        GLint success;
        glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
        if (!success) {
            GLchar infoLog[512];
            glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
            std::cerr << "Shader program linking error: " << infoLog << std::endl;
        }

        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        // Create VAO, VBO, EBO for text rendering
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);

        // Set up vertex attributes
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
        glEnableVertexAttribArray(0);

        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(sizeof(glm::vec3)));
        glEnableVertexAttribArray(1);

        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(sizeof(glm::vec3) + sizeof(glm::vec2)));
        glEnableVertexAttribArray(2);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);

        // Set up projection matrix
        setProjection(windowWidth, windowHeight);

        calculateCharacterWidths();
    }

    ~TextRenderer() {
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &EBO);
        glDeleteVertexArrays(1, &VAO);
        glDeleteProgram(shaderProgram);
        glDeleteTextures(1, &atlas.textureID);
    }

    void setProjection(int windowWidth, int windowHeight) {
        projection = glm::ortho(0.0f, (float)windowWidth, (float)windowHeight, 0.0f, -1.0f, 1.0f);
    }


    void renderText(const std::string& text, float x, float y, float scale, glm::vec4 color, bool centered = false) {
        glUseProgram(shaderProgram);

        // Set uniforms
        GLuint projLoc = glGetUniformLocation(shaderProgram, "projection");
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));

        GLuint fontTexLoc = glGetUniformLocation(shaderProgram, "fontTexture");
        glUniform1i(fontTexLoc, 0);

        GLuint useColorLoc = glGetUniformLocation(shaderProgram, "useColor");
        glUniform1i(useColorLoc, 0); // Set to 1 if your font atlas is colored

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, atlas.textureID);

        glBindVertexArray(VAO);

        // If text should be centered, calculate the starting position
        if (centered) {
            float textWidth = 0;
            for (char c : text) {
                // Use the calculated width for each character
                float charWidth = static_cast<float>(charWidths[c]);// charWidths.count(c) ? charWidths[c] : atlas.charWidth / 2;
                textWidth += (8 + charWidth) * scale;
            }
            x = windowWidth / 2.0f - textWidth / 2.0f;
        }

        // For each character, create a quad with appropriate texture coordinates
        std::vector<Vertex> vertices;
        std::vector<GLuint> indices;

        float xpos = x;
        float ypos = y;
        GLuint indexOffset = 0;

        for (char c : text) {
            // Get ASCII value of the character
            unsigned char charValue = static_cast<unsigned char>(c);

            // Calculate position in the atlas using ASCII value
            int atlasX = (charValue % atlas.charsPerRow) * atlas.charWidth;
            int atlasY = (charValue / atlas.charsPerRow) * atlas.charHeight;

            // Calculate texture coordinates
            float texLeft = atlasX / (float)atlas.atlasWidth;
            float texRight = (atlasX + atlas.charWidth) / (float)atlas.atlasWidth;
            float texTop = atlasY / (float)atlas.atlasHeight;
            float texBottom = (atlasY + atlas.charHeight) / (float)atlas.atlasHeight;

            // Get the character's calculated width
            float charWidth = static_cast<float>(charWidths[charValue]);// charWidths.count(charValue) ? charWidths[charValue] : atlas.charWidth / 2;

            // Calculate quad vertices
            float quadLeft = xpos;
            float quadRight = xpos + atlas.charWidth * scale; // Use full cell width for texture
            float quadTop = ypos;
            float quadBottom = ypos + atlas.charHeight * scale;

            // Add vertices
            vertices.push_back({ {quadLeft, quadTop, 0.0f}, {texLeft, texTop}, color });
            vertices.push_back({ {quadRight, quadTop, 0.0f}, {texRight, texTop}, color });
            vertices.push_back({ {quadRight, quadBottom, 0.0f}, {texRight, texBottom}, color });
            vertices.push_back({ {quadLeft, quadBottom, 0.0f}, {texLeft, texBottom}, color });

            // Add indices
            indices.push_back(indexOffset + 0);
            indices.push_back(indexOffset + 1);
            indices.push_back(indexOffset + 2);
            indices.push_back(indexOffset + 0);
            indices.push_back(indexOffset + 2);
            indices.push_back(indexOffset + 3);

            indexOffset += 4;

            // Advance cursor using the calculated width
            // add 8 pixels of padding between characters
            xpos += (8 + charWidth) * scale;
        }

        // Upload vertex and index data
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_DYNAMIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_DYNAMIC_DRAW);

        // Draw text
        glm::mat4 model = glm::mat4(1.0f);
        GLuint modelLoc = glGetUniformLocation(shaderProgram, "model");
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

        // Enable blending for transparent font
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glDrawElements(GL_TRIANGLES, (GLsizei)indices.size(), GL_UNSIGNED_INT, 0);

        // Reset state
        glDisable(GL_BLEND);
        glBindVertexArray(0);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
};



TextRenderer* textRenderer = nullptr;


void displayFPS() 
{
    static int frame_count = 0;
    static float lastTime = 0.0f;
    static float fps = 0.0f;

    frame_count++;

    float currentTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
    float deltaTime = currentTime - lastTime;

    if (deltaTime >= 1.0f) 
    {
        fps = frame_count / deltaTime;
        frame_count = 0;
        lastTime = currentTime;
    }

    std::string fpsText = "FPS: " + std::to_string(static_cast<int>(fps));

    if(textRenderer)
    textRenderer->renderText(fpsText, 0.0, 10, 0.5f, glm::vec4(1.0f, 1.0f, 1.0f, 1.0f), true);
}









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

// Obstacle stamp shader - uses a sprite texture as a stamp
const char* obstacleStampFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D obstacles;      // Current obstacle field
uniform sampler2D stampTexture;   // Sprite texture to stamp
uniform vec2 stampCenter;         // Center position in normalized coords [0,1]
uniform vec2 stampHalfSize;       // Half-size of stamp in normalized coords (already aspect-corrected)
uniform float threshold;          // Alpha/intensity threshold for stamp (default 0.5)
uniform float addOrRemove;        // 1.0 to add obstacles, 0.0 to remove
uniform int useAlpha;             // 1 = use alpha channel, 0 = use red/luminance

void main() {
    float current = texture(obstacles, texCoord).x;
    
    // Calculate position relative to stamp center in normalized texture space
    vec2 diff = texCoord - stampCenter;
    
    // Check if we're within the stamp bounds
    // stampHalfSize is pre-computed to account for aspect ratio
    vec2 relPos = diff / (stampHalfSize * 2.0) + 0.5;
    
    if (relPos.x >= 0.0 && relPos.x <= 1.0 && relPos.y >= 0.0 && relPos.y <= 1.0) {
        // Sample the stamp texture (flip Y for typical image coordinates)
        vec2 stampUV = vec2(relPos.x, 1.0 - relPos.y);
        vec4 stampSample = texture(stampTexture, stampUV);
        
        // Get stamp value based on mode
        float stampValue;
        if (useAlpha == 1) {
            stampValue = stampSample.a;
        } else {
            // Use luminance of RGB
            stampValue = dot(stampSample.rgb, vec3(0.299, 0.587, 0.114));
        }
        
        // Apply threshold
        if (stampValue >= threshold) {
            if (addOrRemove > 0.5) {
                // Add mode: set obstacle where stamp is solid
                fragColor = vec4(1.0, 0.0, 0.0, 1.0);
            } else {
                // Remove mode: clear obstacle where stamp is solid
                fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            }
        } else {
            // Below threshold, keep current value
            fragColor = vec4(current, 0.0, 0.0, 1.0);
        }
    } else {
        // Outside stamp bounds, keep current value
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
uniform sampler2D background;

uniform vec2 texelSize;
uniform float time;


float WIDTH = texelSize.x;
float HEIGHT = texelSize.y;
float aspect_ratio = WIDTH/HEIGHT;



vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() 
{
    float obstacle = texture(obstacles, texCoord).x;
    
    if (obstacle > 0.5) 
    {
        fragColor = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }


    vec2 adjustedCoord = texCoord;
    
    // For non-square textures, adjust sampling to prevent stretching
    if (1.0/aspect_ratio > 1.0) {
        adjustedCoord.x = (adjustedCoord.x - 0.5) * aspect_ratio + 0.5;
    } else if (1.0/aspect_ratio < 1.0) {
        adjustedCoord.y = (adjustedCoord.y - 0.5) / aspect_ratio + 0.5;
    }

    vec2 scrolledCoord = adjustedCoord;
    scrolledCoord.x += time * 0.01;

    vec4 bgColor = texture(background, scrolledCoord);









  
    // Get density and colors at adjusted position
    float redIntensity = texture(density, texCoord).x;
    float blueIntensity = texture(density, texCoord).y;

    float d = redIntensity + blueIntensity;

    // Create color vectors based on intensity
    vec4 redFluidColor = vec4(redIntensity, 0.0, 0.0, redIntensity);
    vec4 blueFluidColor = vec4(0.0, 0.0, blueIntensity, blueIntensity);

    // Combine both colors
    vec4 combinedColor = redFluidColor + blueFluidColor;

    vec4 blendedBackground = bgColor;//vec4(0.0, 0.0, 0.0, 0.0);

    vec4 color1 = blendedBackground;
    vec4 color2 = vec4(0.0, 0.125, 0.25, 1.0);
    vec4 color3 = combinedColor;
    vec4 color4 = vec4(0.0, 0.0, 0.0, 1.0);

    if(length(redFluidColor.r) > 0.5)
        color4 = vec4(0.0, 0.0, 0.0, 0.0);
    else
        color4 = vec4(1.0, 1.0, 1.0, 0.0);

    // toon shading:
    if (d < 0.25) {
        fragColor = color1;
    } else if (d < 0.5) {
        fragColor = color2;
    } else if (d < 0.75) {
        fragColor = color3;
    } else {
        fragColor = color4;
    }
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

// Sprite drawing shader - draws a texture at a specific screen position
const char* spriteVertexSource = R"(
#version 400 core
layout(location = 0) in vec2 position;
out vec2 texCoord;

uniform vec2 spritePos;      // Position in normalized device coords [-1, 1]
uniform vec2 spriteSize;     // Size in normalized device coords

void main() {
    // position is in [-1, 1] range for the quad
    // Map to sprite position and size
    vec2 pos = spritePos + (position * 0.5 + 0.5) * spriteSize;
    texCoord = position * 0.5 + 0.5;
    // Flip texCoord.y for correct image orientation (top-left origin for images)
    texCoord.y = 1.0 - texCoord.y;
    gl_Position = vec4(pos, 0.0, 1.0);
}
)";

const char* spriteFragmentSource = R"(
#version 400 core
in vec2 texCoord;
out vec4 fragColor;

uniform sampler2D spriteTexture;

void main() {
    vec4 color = texture(spriteTexture, texCoord);
    // Discard fully transparent pixels
    if (color.a < 0.01) discard;
    fragColor = color;
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
    obstacleStampProgram = createProgram(vertexShaderSource, obstacleStampFragmentSource);
    copyProgram = createProgram(vertexShaderSource, copyFragmentSource);
    spriteProgram = createProgram(spriteVertexSource, spriteFragmentSource);
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
        createTexture(densityTex[i], SIM_WIDTH, SIM_HEIGHT, GL_RG32F, GL_RG, GL_FLOAT);
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
    glUniform1f(glGetUniformLocation(advectProgram, "dt"), DT * 100.0f);
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
    glUniform1f(glGetUniformLocation(vorticityForceProgram, "dt"), DT);
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

/**
 * Add or remove obstacles using a sprite texture as a stamp.
 *
 * @param stampTexture  OpenGL texture ID of the sprite to use as stamp
 * @param pixelX        X position of stamp's top-left corner in simulation pixels
 * @param pixelY        Y position of stamp's top-left corner in simulation pixels
 *                      (0,0) is the top-left corner of the simulation
 * @param pixelWidth    Width of the stamp in simulation pixels
 * @param pixelHeight   Height of the stamp in simulation pixels
 * @param add           true to add obstacles, false to remove them
 * @param threshold     Alpha/intensity threshold for considering a pixel solid (default 0.5)
 * @param useAlpha      true to use alpha channel, false to use luminance of RGB
 *
 * The coordinate system uses top-left as origin (0,0), with X increasing to the right
 * and Y increasing downward, matching typical image/screen coordinates.
 *
 * The stamp will appear with the correct aspect ratio regardless of the
 * simulation's aspect ratio. For example, a 100x100 pixel stamp will appear
 * as a square even on a non-square simulation grid.
 *
 * Example usage:
 *   // Load a sprite texture (e.g., a star shape with transparency)
 *   GLuint starTex = loadTexture("star.png");
 *
 *   // Stamp a 64x64 pixel star at position (100, 50) from top-left
 *   addObstacleStamp(starTex, 100, 50, 64, 64, true, 0.5f, true);
 *
 *   // Stamp a 100x50 pixel rectangle at the top-left corner
 *   addObstacleStamp(rectTex, 0, 0, 100, 50, true, 0.5f, true);
 *
 *   // Remove obstacles using a 32x32 pixel circular eraser at (200, 150)
 *   addObstacleStamp(circleTex, 200, 150, 32, 32, false, 0.5f, true);
 */
void addObstacleStamp(GLuint stampTexture, int pixelX, int pixelY,
    int pixelWidth, int pixelHeight, bool add,
    float threshold = 0.5f, bool useAlpha = true) {
    glBindFramebuffer(GL_FRAMEBUFFER, tempFBO);
    glViewport(0, 0, SIM_WIDTH, SIM_HEIGHT);

    glUseProgram(obstacleStampProgram);

    // Bind textures
    setTextureUniform(obstacleStampProgram, "obstacles", 0, obstacleTex);
    setTextureUniform(obstacleStampProgram, "stampTexture", 1, stampTexture);

    // Convert pixel coordinates to normalized texture coordinates
    // Input: top-left origin (0,0), Y increases downward
    // OpenGL texture coords: bottom-left origin (0,0), Y increases upward

    // Calculate center position from top-left corner
    // Center in pixel coords (top-left origin)
    float centerPixelX = pixelX + pixelWidth / 2.0f;
    float centerPixelY = pixelY + pixelHeight / 2.0f;

    // Convert to normalized coords [0,1] with bottom-left origin
    // Flip Y: OpenGL's Y=0 is at bottom, our input Y=0 is at top
    float centerNormX = centerPixelX / (float)SIM_WIDTH;
    float centerNormY = 1.0f - (centerPixelY / (float)SIM_HEIGHT);

    // Half-size in normalized coords
    float halfSizeX = (float)pixelWidth / (2.0f * SIM_WIDTH);
    float halfSizeY = (float)pixelHeight / (2.0f * SIM_HEIGHT);

    // Set uniforms
    glUniform2f(glGetUniformLocation(obstacleStampProgram, "stampCenter"), centerNormX, centerNormY);
    glUniform2f(glGetUniformLocation(obstacleStampProgram, "stampHalfSize"), halfSizeX, halfSizeY);
    glUniform1f(glGetUniformLocation(obstacleStampProgram, "threshold"), threshold);
    glUniform1f(glGetUniformLocation(obstacleStampProgram, "addOrRemove"), add ? 1.0f : 0.0f);
    glUniform1i(glGetUniformLocation(obstacleStampProgram, "useAlpha"), useAlpha ? 1 : 0);

    drawQuad();

    // Copy back to obstacle texture
    glBindFramebuffer(GL_FRAMEBUFFER, obstacleFBO);
    glUseProgram(copyProgram);
    setTextureUniform(copyProgram, "source", 0, tempTex);
    drawQuad();
}

/**
 * Helper function to load a texture from raw RGBA pixel data.
 * Useful for creating stamp textures programmatically or from loaded images.
 *
 * @param data    Pointer to RGBA pixel data (4 bytes per pixel)
 * @param width   Width of the texture in pixels
 * @param height  Height of the texture in pixels
 * @return        OpenGL texture ID
 */
GLuint createStampTextureFromData(const unsigned char* data, int width, int height) {
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return tex;
}

/**
 * Create a circular stamp texture programmatically.
 *
 * @param radius  Radius in pixels
 * @return        OpenGL texture ID of a circle stamp
 */
GLuint createCircleStamp(int radius) {
    int size = radius * 2;
    std::vector<unsigned char> data(size * size * 4, 0);

    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            float dx = x - radius + 0.5f;
            float dy = y - radius + 0.5f;
            float dist = sqrtf(dx * dx + dy * dy);

            int idx = (y * size + x) * 4;
            if (dist <= radius) {
                data[idx + 0] = 255;  // R
                data[idx + 1] = 255;  // G
                data[idx + 2] = 255;  // B
                data[idx + 3] = 255;  // A
            }
        }
    }

    return createStampTextureFromData(data.data(), size, size);
}

/**
 * Create a star-shaped stamp texture programmatically.
 *
 * @param outerRadius  Outer radius of star points in pixels
 * @param innerRadius  Inner radius (between points) in pixels
 * @param points       Number of star points
 * @return             OpenGL texture ID of a star stamp
 */
GLuint createStarStamp(int outerRadius, int innerRadius, int points) {
    int size = outerRadius * 2;
    std::vector<unsigned char> data(size * size * 4, 0);

    float angleStep = 3.14159265f / points;

    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            float dx = x - outerRadius + 0.5f;
            float dy = y - outerRadius + 0.5f;
            float dist = sqrtf(dx * dx + dy * dy);
            float angle = atan2f(dy, dx);
            if (angle < 0) angle += 2.0f * 3.14159265f;

            // Calculate the radius at this angle for the star shape
            int sector = (int)(angle / angleStep);
            float localAngle = fmodf(angle, angleStep * 2.0f);
            if (localAngle > angleStep) localAngle = angleStep * 2.0f - localAngle;

            float starRadius = innerRadius + (outerRadius - innerRadius) * (1.0f - localAngle / angleStep);

            int idx = (y * size + x) * 4;
            if (dist <= starRadius) {
                data[idx + 0] = 255;
                data[idx + 1] = 255;
                data[idx + 2] = 255;
                data[idx + 3] = 255;
            }
        }
    }

    return createStampTextureFromData(data.data(), size, size);
}

/**
 * Create a rectangular stamp texture.
 *
 * @param width   Width in pixels
 * @param height  Height in pixels
 * @return        OpenGL texture ID of a rectangle stamp
 */
GLuint createRectangleStamp(int width, int height) {
    std::vector<unsigned char> data(width * height * 4, 255);  // All white, fully opaque
    return createStampTextureFromData(data.data(), width, height);
}

/**
 * Load a texture from an image file using stb_image.
 * Supports PNG, JPG, BMP, TGA, and other common formats.
 *
 * @param filename    Path to the image file
 * @param outWidth    Output parameter for the image width in pixels
 * @param outHeight   Output parameter for the image height in pixels
 * @return            OpenGL texture ID, or 0 if loading failed
 */
GLuint loadTextureFromFile(const char* filename, int* outWidth, int* outHeight) {
    int width, height, channels;

    // stb_image loads with (0,0) at top-left, which matches our coordinate system
    stbi_set_flip_vertically_on_load(0);  // Don't flip - we handle it in the shader

    unsigned char* data = stbi_load(filename, &width, &height, &channels, 4);  // Force RGBA
    if (!data) {
        std::cerr << "Failed to load texture: " << filename << std::endl;
        std::cerr << "stb_image error: " << stbi_failure_reason() << std::endl;
        if (outWidth) *outWidth = 0;
        if (outHeight) *outHeight = 0;
        return 0;
    }

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    stbi_image_free(data);

    if (outWidth) *outWidth = width;
    if (outHeight) *outHeight = height;

    std::cout << "Loaded texture: " << filename << " (" << width << "x" << height << ")" << std::endl;
    return tex;
}

/**
 * Draw a texture to the screen using pixel-based coordinates.
 *
 * @param texture     OpenGL texture ID of the sprite to draw
 * @param pixelX      X position of the sprite's top-left corner in window pixels
 * @param pixelY      Y position of the sprite's top-left corner in window pixels
 *                    (0,0) is the top-left corner of the window
 * @param pixelWidth  Width to draw the sprite in window pixels
 * @param pixelHeight Height to draw the sprite in window pixels
 *
 * The coordinate system uses top-left as origin (0,0), with X increasing to the right
 * and Y increasing downward, matching typical screen/image coordinates.
 *
 * Example usage:
 *   // Draw a sprite at position (100, 50) with size 64x64 pixels
 *   drawSprite(myTexture, 100, 50, 64, 64);
 *
 *   // Draw a sprite at the top-left corner of the window
 *   drawSprite(myTexture, 0, 0, spriteWidth, spriteHeight);
 *
 *   // Draw a sprite scaled to 2x its original size
 *   drawSprite(myTexture, 200, 100, originalWidth * 2, originalHeight * 2);
 */
void drawSprite(GLuint texture, int pixelX, int pixelY, int pixelWidth, int pixelHeight) {
    if (texture == 0) return;

    // Enable blending for transparency
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(spriteProgram);

    // Convert pixel coordinates to normalized device coordinates [-1, 1]
    // Window coordinate system: (0,0) at top-left, Y increases downward
    // NDC: (-1,-1) at bottom-left, (1,1) at top-right

    // Convert top-left corner position from pixel coords to NDC
    // pixelX=0 -> ndcX=-1, pixelX=windowWidth -> ndcX=1
    // pixelY=0 -> ndcY=1 (top), pixelY=windowHeight -> ndcY=-1 (bottom)
    float ndcX = (2.0f * pixelX / windowWidth) - 1.0f;
    float ndcY = 1.0f - (2.0f * pixelY / windowHeight);  // Flip Y

    // Convert size from pixels to NDC units
    float ndcWidth = 2.0f * pixelWidth / windowWidth;
    float ndcHeight = 2.0f * pixelHeight / windowHeight;

    // The sprite shader expects position as top-left corner in NDC
    // and size as the full width/height in NDC
    // Adjust Y position since we draw from bottom-left of the sprite quad
    float spritePosX = ndcX;
    float spritePosY = ndcY - ndcHeight;  // Move down by height (NDC Y is flipped)

    glUniform2f(glGetUniformLocation(spriteProgram, "spritePos"), spritePosX, spritePosY);
    glUniform2f(glGetUniformLocation(spriteProgram, "spriteSize"), ndcWidth, ndcHeight);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glUniform1i(glGetUniformLocation(spriteProgram, "spriteTexture"), 0);

    drawQuad();

    glDisable(GL_BLEND);


}

void simulate() 
{
    GLuint clearColor[4] = { 0, 0, 0, 0 };
    glClearTexImage(obstacleTex, 0, GL_RGBA, GL_UNSIGNED_BYTE, clearColor);

    addObstacleStamp(protagonistTex, 100, 100,
        protagonistWidth, protagonistHeight, true,
        0.5f, true);

    // Advect velocity
    advect(velocityTex[currentVelocity], velocityTex[currentVelocity], velocityFBO[1 - currentVelocity], VELOCITY_DISSIPATION);
    currentVelocity = 1 - currentVelocity;

    // Advect density
    advect(velocityTex[currentVelocity], densityTex[currentDensity], densityFBO[1 - currentDensity], DENSITY_DISSIPATION);
    currentDensity = 1 - currentDensity;

    // Vorticity confinement
    if (1/*vorticityEnabled*/) {
        computeVorticity();
        applyVorticityForce();
    }

    // Pressure projection
    computeDivergence();
    clearPressure();
    jacobi();
    subtractGradient();
}

void display()
{
    //simulate();


    // Variable time step
    //static float lastTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0f; // Convert to seconds
    //float currentTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

    //const double d = 1.0 / FPS;

    //DT = currentTime - lastTime;

    //if (DT > d)
    //{
    //    simulate();
    //    GLOBAL_TIME += DT;
    //    lastTime = currentTime;
    //}





	// Fixed time step
	static double currentTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
	static double accumulator = 0.0;

	double newTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
	double frameTime = newTime - currentTime;
	currentTime = newTime;

	if (frameTime > DT)
		frameTime = DT;

	accumulator += frameTime;

	while (accumulator >= DT)
	{
		simulate();
		accumulator -= DT;
		GLOBAL_TIME += DT;
	}






    // Add continuous sources based on mouse input
    if (leftMouseDown && !shiftDown) {
        float x = (float)mouseX / windowWidth;
        float y = 1.0f - (float)mouseY / windowHeight;

        if(red_mode)
            addSource(densityTex, densityFBO, currentDensity, x, y, 1, 0, 0, 0.0008f);
        else
            addSource(densityTex, densityFBO, currentDensity, x, y, 0, 1, 0, 0.0008f);
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
    setTextureUniform(displayProgram, "background", 3, backgroundTex);
    glUniform1f(glGetUniformLocation(displayProgram, "time"), GLOBAL_TIME);
    glUniform2f(glGetUniformLocation(displayProgram, "texelSize"), 1.0f / windowWidth, 1.0f / windowHeight);


    drawQuad();

    // Draw protagonist sprite on top of the fluid simulation
    // Draw at position (100, 100) with its original size
    if (protagonistTex != 0) {
        drawSprite(protagonistTex, 100, 100, protagonistWidth, protagonistHeight);
    }

    //if (foregroundTex != 0) {
    //    drawSprite(foregroundTex, -100, 0, foregroundWidth, foregroundHeight);
    //}

    displayFPS();


    glutSwapBuffers();
    glutPostRedisplay();
}

void reshape(int w, int h) {
    windowWidth = w;
    windowHeight = h;
}

void keyboard(unsigned char key, int x, int y) 
{
    switch (key) 
    {

    case 'x':
    {
        red_mode = !red_mode;
        break;
    }

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
    std::cout << "  Middle mouse / Shift+Left: Add obstacles (brush)" << std::endl;
    std::cout << "  'r': Reset simulation" << std::endl;
    std::cout << "  'o': Clear obstacles" << std::endl;
    std::cout << "  '+'/'-': Adjust vorticity strength" << std::endl;
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


    textRenderer = new TextRenderer("media/font.png", windowWidth, windowHeight);

    // Initialize
    initShaders();
    initTextures();
    initQuad();

    // Load protagonist texture
    protagonistTex = loadTextureFromFile("media/protagonist.png", &protagonistWidth, &protagonistHeight);
    if (protagonistTex == 0) {
        std::cout << "Warning: Could not load protagonist.png - sprite drawing will be disabled" << std::endl;
   
        return 1;
    }

    foregroundTex = loadTextureFromFile("media/foreground.png", &foregroundWidth, &foregroundHeight);
    if (foregroundTex == 0) {
        std::cout << "Warning: Could not load foreground.png - sprite drawing will be disabled" << std::endl;
  
        return 2;
    }

    backgroundTex = loadTextureFromFile("media/background.png", &backgroundWidth, &backgroundHeight);
    if (backgroundTex == 0) {
        std::cout << "Warning: Could not load background.png - sprite drawing will be disabled" << std::endl;

        return 3;
    }


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
