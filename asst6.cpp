////////////////////////////////////////////////////////////////////////
//
//   Harvard University
//   CS175 : Computer Graphics
//   Professor Steven Gortler
//
////////////////////////////////////////////////////////////////////////

#include <cstddef>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <list>

#ifdef __MAC__
#   include <OpenGL/gl3.h>
#   include <GLUT/glut.h>
#else
#   include <GL/glew.h>
#   include <GL/glut.h>
#endif

#include "ppm.h"
#include "cvec.h"
#include "matrix4.h"
#include "rigtform.h"
#include "glsupport.h"
#include "geometrymaker.h"
#include "arcball.h"
#include "scenegraph.h"

#include "asstcommon.h"
#include "drawer.h"
#include "picker.h"
#include "sgutils.h"

using namespace std;

// G L O B A L S ///////////////////////////////////////////////////

// --------- IMPORTANT --------------------------------------------------------
// Before you start working on this assignment, set the following variable
// properly to indicate whether you want to use OpenGL 2.x with GLSL 1.0 or
// OpenGL 3.x+ with GLSL 1.5.
//
// Set g_Gl2Compatible = true to use GLSL 1.0 and g_Gl2Compatible = false to
// use GLSL 1.5. Use GLSL 1.5 unless your system does not support it.
//
// If g_Gl2Compatible=true, shaders with -gl2 suffix will be loaded.
// If g_Gl2Compatible=false, shaders with -gl3 suffix will be loaded.
// To complete the assignment you only need to edit the shader files that get
// loaded
// ----------------------------------------------------------------------------
const bool g_Gl2Compatible = false;


static const float g_frustMinFov = 60.0;  // A minimal of 60 degree field of view
static float g_frustFovY = g_frustMinFov; // FOV in y direction (updated by updateFrustFovY)

static const float g_frustNear = -0.1;    // near plane
static const float g_frustFar = -50.0;    // far plane
static const float g_groundY = -2.0;      // y coordinate of the ground
static const float g_groundSize = 10.0;   // half the ground length

enum SkyMode {WORLD_SKY=0, SKY_SKY=1};

static int g_windowWidth = 512;
static int g_windowHeight = 512;
static bool g_mouseClickDown = false;    // is the mouse button pressed
static bool g_mouseLClickButton, g_mouseRClickButton, g_mouseMClickButton;
static bool g_spaceDown = false;         // space state, for middle mouse emulation
static int g_mouseClickX, g_mouseClickY; // coordinates for mouse click event
static int g_activeShader = 0;

static SkyMode g_activeCameraFrame = WORLD_SKY;

static bool g_displayArcball = true;
static double g_arcballScreenRadius = 100; // number of pixels
static double g_arcballScale = 1;

static bool g_pickingMode = false;
static bool g_animMode = false;

// -------- Shaders

static const int g_numShaders = 3, g_numRegularShaders = 2;
static const int PICKING_SHADER = 2;
static const char * const g_shaderFiles[g_numShaders][2] = {
  {"./shaders/basic-gl3.vshader", "./shaders/diffuse-gl3.fshader"},
  {"./shaders/basic-gl3.vshader", "./shaders/solid-gl3.fshader"},
  {"./shaders/basic-gl3.vshader", "./shaders/pick-gl3.fshader"}
};
static const char * const g_shaderFilesGl2[g_numShaders][2] = {
  {"./shaders/basic-gl2.vshader", "./shaders/diffuse-gl2.fshader"},
  {"./shaders/basic-gl2.vshader", "./shaders/solid-gl2.fshader"},
  {"./shaders/basic-gl2.vshader", "./shaders/pick-gl2.fshader"}
};
static vector<shared_ptr<ShaderState> > g_shaderStates; // our global shader states


// --------- Geometry

// Macro used to obtain relative offset of a field within a struct
#define FIELD_OFFSET(StructType, field) ((GLvoid*)offsetof(StructType, field))

// A vertex with floating point position and normal
struct VertexPN {
  Cvec3f p, n;

  VertexPN() {}
  VertexPN(float x, float y, float z,
           float nx, float ny, float nz)
    : p(x,y,z), n(nx, ny, nz)
  {}

  // Define copy constructor and assignment operator from GenericVertex so we can
  // use make* functions from geometrymaker.h
  VertexPN(const GenericVertex& v) {
    *this = v;
  }

  VertexPN& operator = (const GenericVertex& v) {
    p = v.pos;
    n = v.normal;
    return *this;
  }
};

struct Geometry {
  GlBufferObject vbo, ibo;
  GlArrayObject vao;
  int vboLen, iboLen;

  Geometry(VertexPN *vtx, unsigned short *idx, int vboLen, int iboLen) {
    this->vboLen = vboLen;
    this->iboLen = iboLen;

    // Now create the VBO and IBO
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(VertexPN) * vboLen, vtx, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned short) * iboLen, idx, GL_STATIC_DRAW);
  }

  void draw(const ShaderState& curSS) {
    // bind the object's VAO
    glBindVertexArray(vao);

    // Enable the attributes used by our shader
    safe_glEnableVertexAttribArray(curSS.h_aPosition);
    safe_glEnableVertexAttribArray(curSS.h_aNormal);

    // bind vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    safe_glVertexAttribPointer(curSS.h_aPosition, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPN), FIELD_OFFSET(VertexPN, p));
    safe_glVertexAttribPointer(curSS.h_aNormal, 3, GL_FLOAT, GL_FALSE, sizeof(VertexPN), FIELD_OFFSET(VertexPN, n));

    // bind ibo
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);

    // draw!
    glDrawElements(GL_TRIANGLES, iboLen, GL_UNSIGNED_SHORT, 0);

    // Disable the attributes used by our shader
    safe_glDisableVertexAttribArray(curSS.h_aPosition);
    safe_glDisableVertexAttribArray(curSS.h_aNormal);

    // disable VAO
    glBindVertexArray(NULL);
  }
};

typedef SgGeometryShapeNode<Geometry> MyShapeNode;

// Vertex buffer and index buffer associated with the ground and cube geometry
static shared_ptr<Geometry> g_ground, g_cube, g_sphere;

// --------- Scene

static const Cvec3 g_light1(2.0, 3.0, 14.0), g_light2(-2, -3.0, -5.0);  // define two lights positions in world space

static shared_ptr<SgRootNode> g_world;
static shared_ptr<SgRbtNode> g_skyNode, g_groundNode, g_robot2Node;
static shared_ptr<SgObjectNode> g_robot1Node;

static shared_ptr<SgRbtNode> g_currentCameraNode;
static shared_ptr<SgRbtNode> g_currentPickedRbtNode;
static list<vector<RigTForm>> g_script;
static list<vector<RigTForm>>::iterator g_currentFrame = g_script.begin();
static vector<shared_ptr<SgRbtNode>> g_map;
static bool firstLeft = true;
static int g_msBetweenKeyFrames = 2000;
static int g_animateFramesPerSecond = 60;
static bool g_loopState = false;
static bool g_physics_mode = true;
static vector<shared_ptr<SgObjectNode>> g_objectNodes;


///////////////// END OF G L O B A L S //////////////////////////////////////////////////

static void initGround() {
  // A x-z plane at y = g_groundY of dimension [-g_groundSize, g_groundSize]^2
  VertexPN vtx[4] = {
    VertexPN(-g_groundSize, g_groundY, -g_groundSize, 0, 1, 0),
    VertexPN(-g_groundSize, g_groundY,  g_groundSize, 0, 1, 0),
    VertexPN( g_groundSize, g_groundY,  g_groundSize, 0, 1, 0),
    VertexPN( g_groundSize, g_groundY, -g_groundSize, 0, 1, 0)
  };
  unsigned short idx[] = {0, 1, 2, 0, 2, 3};
  g_ground.reset(new Geometry(&vtx[0], &idx[0], 4, 6));
}



static void initCubes() {
  int ibLen, vbLen;
  getCubeVbIbLen(vbLen, ibLen);


  // Temporary storage for cube geometry
  vector<VertexPN> vtx(vbLen);
  vector<unsigned short> idx(ibLen);

  makeCube(1, vtx.begin(), idx.begin());
  g_cube.reset(new Geometry(&vtx[0], &idx[0], vbLen, ibLen));
}

static void initSphere() {
  int ibLen, vbLen;
  getSphereVbIbLen(20, 10, vbLen, ibLen);

  // Temporary storage for sphere geometry
  vector<VertexPN> vtx(vbLen);
  vector<unsigned short> idx(ibLen);
  makeSphere(1, 20, 10, vtx.begin(), idx.begin());
  g_sphere.reset(new Geometry(&vtx[0], &idx[0], vtx.size(), idx.size()));
}

static void initRobots() {
  // Init whatever geometry needed for the robots
}

// takes a projection matrix and send to the the shaders
inline void sendProjectionMatrix(const ShaderState& curSS, const Matrix4& projMatrix) {
  GLfloat glmatrix[16];
  projMatrix.writeToColumnMajorMatrix(glmatrix); // send projection matrix
  safe_glUniformMatrix4fv(curSS.h_uProjMatrix, glmatrix);
}

// update g_frustFovY from g_frustMinFov, g_windowWidth, and g_windowHeight
static void updateFrustFovY() {
  if (g_windowWidth >= g_windowHeight)
    g_frustFovY = g_frustMinFov;
  else {
    const double RAD_PER_DEG = 0.5 * CS175_PI/180;
    g_frustFovY = atan2(sin(g_frustMinFov * RAD_PER_DEG) * g_windowHeight / g_windowWidth, cos(g_frustMinFov * RAD_PER_DEG)) / RAD_PER_DEG;
  }
}

static Matrix4 makeProjectionMatrix() {
  return Matrix4::makeProjection(
           g_frustFovY, g_windowWidth / static_cast <double> (g_windowHeight),
           g_frustNear, g_frustFar);
}

enum ManipMode {
  ARCBALL_ON_PICKED,
  ARCBALL_ON_SKY,
  EGO_MOTION
};

static ManipMode getManipMode() {
  // if nothing is picked or the picked transform is the transfrom we are viewing from
  if (g_currentPickedRbtNode == NULL || g_currentPickedRbtNode == g_currentCameraNode) {
    if (g_currentCameraNode == g_skyNode && g_activeCameraFrame == WORLD_SKY)
      return ARCBALL_ON_SKY;
    else
      return EGO_MOTION;
  }
  else
    return ARCBALL_ON_PICKED;
}

static bool shouldUseArcball() {
  return (g_currentPickedRbtNode!=0);
  //  return getManipMode() != EGO_MOTION;
}

// The translation part of the aux frame either comes from the current
// active object, or is the identity matrix when
static RigTForm getArcballRbt() {
  switch (getManipMode()) {
  case ARCBALL_ON_PICKED:
    return getPathAccumRbt(g_world, g_currentPickedRbtNode);
  case ARCBALL_ON_SKY:
    return RigTForm();
  case EGO_MOTION:
    return getPathAccumRbt(g_world, g_currentCameraNode);
  default:
    throw runtime_error("Invalid ManipMode");
  }
}

static void updateArcballScale() {
  RigTForm arcballEye = inv(getPathAccumRbt(g_world, g_currentCameraNode)) * getArcballRbt();
  double depth = arcballEye.getTranslation()[2];
  if (depth > -CS175_EPS)
    g_arcballScale = 0.02;
  else
g_arcballScale = getScreenToEyeScale(depth, g_frustFovY, g_windowHeight);
}

static void drawArcBall(const ShaderState& curSS) {
    // switch to wire frame mode
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    RigTForm arcballEye = inv(getPathAccumRbt(g_world, g_currentCameraNode)) * getArcballRbt();
    Matrix4 MVM = rigTFormToMatrix(arcballEye) * Matrix4::makeScale(Cvec3(1, 1, 1) * g_arcballScale * g_arcballScreenRadius);
    sendModelViewNormalMatrix(curSS, MVM, normalMatrix(MVM));

    safe_glUniform3f(curSS.h_uColor, 0.27, 0.82, 0.35); // set color
    g_sphere->draw(curSS);

    // switch back to solid mode
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

static void drawStuff(const ShaderState& curSS, bool picking) {
    // if we are not translating, update arcball scale
    if (!(g_mouseMClickButton || (g_mouseLClickButton && g_mouseRClickButton) || (g_mouseLClickButton && !g_mouseRClickButton && g_spaceDown)))
        updateArcballScale();

    // build & send proj. matrix to vshader
    const Matrix4 projmat = makeProjectionMatrix();
    sendProjectionMatrix(curSS, projmat);

    const RigTForm eyeRbt = getPathAccumRbt(g_world, g_currentCameraNode);
    const RigTForm invEyeRbt = inv(eyeRbt);

    const Cvec3 eyeLight1 = Cvec3(invEyeRbt * Cvec4(g_light1, 1));
    const Cvec3 eyeLight2 = Cvec3(invEyeRbt * Cvec4(g_light2, 1));
    safe_glUniform3f(curSS.h_uLight, eyeLight1[0], eyeLight1[1], eyeLight1[2]);
    safe_glUniform3f(curSS.h_uLight2, eyeLight2[0], eyeLight2[1], eyeLight2[2]);

    if (!picking) {
        Drawer drawer(invEyeRbt, curSS);
        g_world->accept(drawer);

        if (g_displayArcball && shouldUseArcball())
            drawArcBall(curSS);
    }
    else {
        Picker picker(invEyeRbt, curSS);
        g_world->accept(picker);
        glFlush();
        g_currentPickedRbtNode = picker.getRbtNodeAtXY(g_mouseClickX, g_mouseClickY);
        if (g_currentPickedRbtNode == g_groundNode)
            g_currentPickedRbtNode = shared_ptr<SgRbtNode>(); // set to NULL

        cout << (g_currentPickedRbtNode ? "Part picked" : "No part picked") << endl;
    }
}

static void display() {
    glUseProgram(g_shaderStates[g_activeShader]->program);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    drawStuff(*g_shaderStates[g_activeShader], false);

    glutSwapBuffers();

    checkGlErrors();
}

static void pick() {
    // We need to set the clear color to black, for pick rendering.
    // so let's save the clear color
    GLdouble clearColor[4];
    glGetDoublev(GL_COLOR_CLEAR_VALUE, clearColor);

    glClearColor(0, 0, 0, 0);

    // using PICKING_SHADER as the shader
    glUseProgram(g_shaderStates[PICKING_SHADER]->program);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    drawStuff(*g_shaderStates[PICKING_SHADER], true);

    // Uncomment below and comment out the glutPostRedisplay in mouse(...) call back
    // to see result of the pick rendering pass
    // glutSwapBuffers();

    //Now set back the clear color
    glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);

    checkGlErrors();
}

static Cvec3 lerp(Cvec3 v1, Cvec3 v2, float a) {
    return v1 * (1.0 - a) + v2 * a;
}

static Quat pow(Quat q, float a) {
    float norm_const = sqrt(q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    float theta_2 = atan2(norm_const, q[0]);
    if (norm_const > 0) {
        float vec_const = sin(a * theta_2) / norm_const;
        return Quat(cos(a * theta_2), vec_const * q[1], vec_const * q[2], vec_const * q[3]);
    }
    return Quat(cos(a * theta_2), 0, 0, 0);
}
static Quat cn(Quat q) {
    if (q[0] < 0){
        return Quat(-q[0], -q[1], -q[2], -q[3]);
    }
    return q;
}
static Quat slerp(Quat q0, Quat q1, float a) {
    return pow(cn(q1 * pow(q0, -1)), a) * q0;
}
static void reshape(const int w, const int h) {
  g_windowWidth = w;
  g_windowHeight = h;
  glViewport(0, 0, w, h);
  cerr << "Size of window is now " << w << "x" << h << endl;
  g_arcballScreenRadius = max(1.0, min(h, w) * 0.25);
  updateFrustFovY();
  glutPostRedisplay();
}

static Cvec3 getArcballDirection(const Cvec2& p, const double r) {
  double n2 = norm2(p);
  if (n2 >= r*r)
    return normalize(Cvec3(p, 0));
  else
    return normalize(Cvec3(p, sqrt(r*r - n2)));
}

static RigTForm moveArcball(const Cvec2& p0, const Cvec2& p1) {
  const Matrix4 projMatrix = makeProjectionMatrix();
  const RigTForm eyeInverse = inv(getPathAccumRbt(g_world, g_currentCameraNode));
  const Cvec3 arcballCenter = getArcballRbt().getTranslation();
  const Cvec3 arcballCenter_ec = Cvec3(eyeInverse * Cvec4(arcballCenter, 1));

  if (arcballCenter_ec[2] > -CS175_EPS)
    return RigTForm();

  Cvec2 ballScreenCenter = getScreenSpaceCoord(arcballCenter_ec,
                                               projMatrix, g_frustNear, g_frustFovY, g_windowWidth, g_windowHeight);
  const Cvec3 v0 = getArcballDirection(p0 - ballScreenCenter, g_arcballScreenRadius);
  const Cvec3 v1 = getArcballDirection(p1 - ballScreenCenter, g_arcballScreenRadius);

  return RigTForm(Quat(0.0, v1[0], v1[1], v1[2]) * Quat(0.0, -v0[0], -v0[1], -v0[2]));
}

static RigTForm doMtoOwrtA(const RigTForm& M, const RigTForm& O, const RigTForm& A) {
  return A * M * inv(A) * O;
}

static RigTForm getMRbt(const double dx, const double dy) {
  RigTForm M;

  if (g_mouseLClickButton && !g_mouseRClickButton && !g_spaceDown) {
    if (shouldUseArcball())
      M = moveArcball(Cvec2(g_mouseClickX, g_mouseClickY), Cvec2(g_mouseClickX + dx, g_mouseClickY + dy));
    else
      M = RigTForm(Quat::makeXRotation(-dy) * Quat::makeYRotation(dx));
  }
  else {
    double movementScale = getManipMode() == EGO_MOTION ? 0.02 : g_arcballScale;
    if (g_mouseRClickButton && !g_mouseLClickButton) {
      M = RigTForm(Cvec3(dx, dy, 0) * movementScale);
    }
    else if (g_mouseMClickButton || (g_mouseLClickButton && g_mouseRClickButton) || (g_mouseLClickButton && g_spaceDown)) {
      M = RigTForm(Cvec3(0, 0, -dy) * movementScale);
    }
  }

  switch (getManipMode()) {
  case ARCBALL_ON_PICKED:
    break;
  case ARCBALL_ON_SKY:
    M = inv(M);
    break;
  case EGO_MOTION:
    //    if (g_mouseLClickButton && !g_mouseRClickButton && !g_spaceDown) // only invert rotation
      M = inv(M);
    break;
  }
  return M;
}

static RigTForm makeMixedFrame(const RigTForm& objRbt, const RigTForm& eyeRbt) {
  return transFact(objRbt) * linFact(eyeRbt);
}

// l = w X Y Z
// o = l O
// a = w A = l (Z Y X)^1 A = l A'
// o = a (A')^-1 O
//   => a M (A')^-1 O = l A' M (A')^-1 O

static void motion(const int x, const int y) {
  if (!g_mouseClickDown || g_animMode)
    return;

  const double dx = x - g_mouseClickX;
  const double dy = g_windowHeight - y - 1 - g_mouseClickY;

  const RigTForm M = getMRbt(dx, dy);   // the "action" matrix

  // the matrix for the auxiliary frame (the w.r.t.)
  RigTForm A = makeMixedFrame(getArcballRbt(), getPathAccumRbt(g_world, g_currentCameraNode));

  shared_ptr<SgRbtNode> target;
  switch (getManipMode()) {
  case ARCBALL_ON_PICKED:
    target = g_currentPickedRbtNode;
    break;
  case ARCBALL_ON_SKY:
    target = g_skyNode;
    break;
  case EGO_MOTION:
    target = g_currentCameraNode;
    break;
  }

  A = inv(getPathAccumRbt(g_world, target, 1)) * A;


  if ((g_mouseLClickButton && !g_mouseRClickButton && !g_spaceDown) //rotating
      && target == g_skyNode) {
    RigTForm My = getMRbt(dx, 0);
    RigTForm Mx = getMRbt(0, dy);
    RigTForm B = makeMixedFrame(getArcballRbt(), RigTForm());
    RigTForm O = doMtoOwrtA(Mx, target->getRbt(), A);
    O = doMtoOwrtA(My, O, B);
    target->setRbt(O);
  }
  else{
    target->setRbt(doMtoOwrtA(M, target->getRbt(), A));
  }
  g_mouseClickX += dx;
  g_mouseClickY += dy;
  glutPostRedisplay();  // we always redraw if we changed the scene
}

static void mouse(const int button, const int state, const int x, const int y) {
  g_mouseClickX = x;
  g_mouseClickY = g_windowHeight - y - 1;  // conversion from GLUT window-coordinate-system to OpenGL window-coordinate-system

  g_mouseLClickButton |= (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN);
  g_mouseRClickButton |= (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN);
  g_mouseMClickButton |= (button == GLUT_MIDDLE_BUTTON && state == GLUT_DOWN);

  g_mouseLClickButton &= !(button == GLUT_LEFT_BUTTON && state == GLUT_UP);
  g_mouseRClickButton &= !(button == GLUT_RIGHT_BUTTON && state == GLUT_UP);
  g_mouseMClickButton &= !(button == GLUT_MIDDLE_BUTTON && state == GLUT_UP);

  g_mouseClickDown = g_mouseLClickButton || g_mouseRClickButton || g_mouseMClickButton;

  if (g_pickingMode && button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
    pick();
    g_pickingMode = false;
    cerr << "Picking mode is off" << endl;
    glutPostRedisplay(); // request redisplay since the arcball will have moved
  }
  glutPostRedisplay();
}

static void keyboardUp(const unsigned char key, const int x, const int y) {
  switch (key) {
  case ' ':
    g_spaceDown = false;
    break;
  }
  glutPostRedisplay();
}

static void load_key() {
    vector<shared_ptr<SgRbtNode>>::iterator map_it = g_map.begin();
    vector<RigTForm>::iterator key_it = (*g_currentFrame).begin();
    while (map_it != g_map.end() && key_it != (*g_currentFrame).end()) {
        (**map_it).setRbt(*key_it);
        ++map_it;
        ++key_it;
    }
}

//A sister to load_key, but it works for any frame passed in.
//Used for 
static void load_frame(vector<RigTForm> frame) {
    vector<shared_ptr<SgRbtNode>>::iterator map_it = g_map.begin();
    vector<RigTForm>::iterator frame_it = frame.begin();
    for (int a = 0; a < 22; a++){
        (**map_it).setRbt(*frame_it);
        ++map_it;
        ++frame_it;
    }
}

static void save_key() {
    vector<shared_ptr<SgRbtNode>>::iterator map_it = g_map.begin();
    vector<RigTForm>::iterator key_it = (*g_currentFrame).begin();
    while (map_it != g_map.end() && key_it != (*g_currentFrame).end()) {
        *key_it = (**map_it).getRbt();
        ++map_it;
        ++key_it;
    }
}

/*static void readFile() {
    ofstream myfile;
    myfile.open("input.txt");
    myfile << "Writing this to a file.\n";
    myfile.close();
}*/
static void writeFile() {
    ofstream output;
    output.open("output.txt");
    output << g_script.size();
    output << ";";
    list<vector<RigTForm>>::iterator dummyIt = g_script.begin();
    output << (*dummyIt).size();
    output << ";";
    output << g_animateFramesPerSecond;
    output << ";";
    output << g_msBetweenKeyFrames;
    output << "\n";
    for (list<vector<RigTForm>>::iterator it1 = g_script.begin(); it1 != g_script.end(); ++it1) {
        for (vector<RigTForm>::iterator it2 = (*it1).begin(); it2 != (*it1).end(); ++it2) {
            Quat q = (*it2).getRotation();
            Cvec3 v = (*it2).getTranslation();
            output << q[0];
            output << " ";
            output << q[1];
            output << " ";
            output << q[2];
            output << " ";
            output << q[3];
            output << " ";
            output << v[0];
            output << " ";
            output << v[1];
            output << " ";
            output << v[2];
            output << "\n";
        }
    }
    output.close();
}

static void readFile() {
    ifstream input;
    input.open("input.txt");
    int numFrames;
    char temp;
    int frameSize;
    input >> numFrames;
    input >> temp;
    input >> frameSize;
    input >> temp;
    input >> g_animateFramesPerSecond;
    input >> temp;
    input >> g_msBetweenKeyFrames;
    list<vector<RigTForm>> new_script;
    for (int a = 0; a < numFrames; a++) {
        vector<RigTForm> frame;
        for (int b = 0; b < frameSize; b++) {
            float rtf[7];
            for (int c = 0; c < 7; c++) {
                input >> rtf[c];
            }
            RigTForm newBoi = RigTForm(Cvec3(rtf[4], rtf[5], rtf[6]), Quat(rtf[0], rtf[1], rtf[2], rtf[3]));
            frame.push_back(newBoi);
        }
        new_script.push_back(frame);
    }
    g_script = new_script;
    g_currentFrame = g_script.begin();
    load_key();

}

static bool interpolateAndDisplay(float t) {
    int end = g_script.size() - 3;
    if ((t >= end && !g_loopState) || g_spaceDown) {
        return true;
    }
    else if (t >= end) {
        while (t >= end) {
            t -= end;
        }
    }
    list<vector<RigTForm>>::iterator it = g_script.begin();
    advance(it, (int)t);
    vector<RigTForm> frame_1 = *it; // _1 is negative 1
    ++it; 
    vector<RigTForm> frame0 = *it;
    ++it;
    vector<RigTForm> frame1 = *it;

    ++it;
    vector<RigTForm> frame2 = *it;
    vector<RigTForm> mixedFrame;
    float a = t - (int)t;

    //Actual Interpolation
    for (int i = 0; i < frame0.size(); i++) {
        Cvec3 c_1 = frame_1[i].getTranslation();
        Cvec3 c0 = frame0[i].getTranslation();
        Cvec3 c1 = frame1[i].getTranslation();
        Cvec3 c2 = frame2[i].getTranslation();
        Cvec3 d =  (c1 - c_1) * .16666666 + c0;
        Cvec3 e = (c2 - c0) * -0.166666666 + c1;
        Cvec3 f =  c0 * (1.0-a) + d*a;
        Cvec3 g = d * (1 - a) + e * a;
        Cvec3 h = e * (1 - a) + c1 * a;
        Cvec3 m = f * (1 - a) + g * a;
        Cvec3 n = g * (1 - a) + h * a;
        Cvec3 interpol_vec = m * (1 - a) + n * a;


        Quat r_1 = frame_1[i].getRotation();
        Quat r0 = frame0[i].getRotation();
        Quat r1 = frame1[i].getRotation();
        Quat r2 = frame2[i].getRotation();
        Quat dq = pow(r1 * pow(r_1, -1), 1.0/6.0)*r0;
        Quat eq = pow(r2 * pow(r0, -1), -1.0 / 6.0) * r1;
        Quat fq = slerp(r0,dq,a);
        Quat gq = slerp(dq, eq, a);
        Quat hq = slerp(eq, r1, a);
        Quat mq = slerp(fq, gq, a);
        Quat nq = slerp(gq, hq, a);
        Quat interpol_quat= slerp(mq, nq, a);
        RigTForm interpol = RigTForm(interpol_vec, interpol_quat);
        mixedFrame.push_back(interpol);
    }
    load_frame(mixedFrame);
    glutPostRedisplay();
    return false;

}

static void physicsTimerCallback(int ms) {
    if (g_physics_mode) {
        for (int a = 0; a < g_objectNodes.size(); a++) {
            (*g_objectNodes[a]).update();
        }
        glutTimerFunc(1000 / g_animateFramesPerSecond,
            physicsTimerCallback,
            ms + 1000 / g_animateFramesPerSecond);
    }
}

static void animateTimerCallback(int ms) {
    g_animMode = true;
    float t = (float)ms / (float)g_msBetweenKeyFrames;

    bool endReached = interpolateAndDisplay(t);
    if (!endReached) {
        glutTimerFunc(1000 / g_animateFramesPerSecond,
            animateTimerCallback,
            ms + 1000 / g_animateFramesPerSecond);
    }
    else{ 
        g_animMode = false;
    }

}
static void move_left() {
    if (distance(g_script.begin(), g_currentFrame) > 1 || (!firstLeft && distance(g_script.begin(), g_currentFrame) > 0)) {
        --g_currentFrame;
        if (firstLeft) {
            --g_currentFrame;
            firstLeft = false;
        }
        load_key();
    }
}
static void move_right() {
    if (distance(g_currentFrame, g_script.end()) > 1) {
        ++g_currentFrame;
        load_key();
    }
}


static void keyboard(const unsigned char key, const int x, const int y) {
    if (g_animMode) {
        if (key == ' ') {
            g_spaceDown = true;
        }
        return;
    }
  switch (key) {
  case '+':
      if (g_msBetweenKeyFrames > (int)(1000 / g_animateFramesPerSecond)) {
          g_msBetweenKeyFrames -= (int)(1000 / g_animateFramesPerSecond);
      }
      if (g_spaceDown && g_msBetweenKeyFrames > 9 * (int)(1000 / g_animateFramesPerSecond)) {
          g_msBetweenKeyFrames -= 9*(int)(1000 / g_animateFramesPerSecond);
      }
      break;
  case '-':
      cout << "\n";
      cout << g_msBetweenKeyFrames;
      cout << " ";
      cout << (int)(1000 / g_animateFramesPerSecond);
      cout << "\n";
      g_msBetweenKeyFrames += (int)(1000 / g_animateFramesPerSecond);
      if (g_spaceDown ) {
          g_msBetweenKeyFrames += 9 * (int)(1000 / g_animateFramesPerSecond);
      }
      break;
  case 'w':
      writeFile();
      break;
  case 'i':
      readFile();
      break;
  case 'r':
      cout << g_script.size();
      break;
  case 'd':
      if (g_script.size() > 1) {
          if (g_currentFrame == g_script.end()) {
              cout << "atEnd\n";
              --g_currentFrame;
              if (firstLeft) {
                  cout << "don't think this happens anymore\n";
                  --g_currentFrame;
                  firstLeft = false;
              }
              g_script.erase(g_currentFrame);
              load_key();
          }
          else {
              list<vector<RigTForm>>::iterator tempFrame = g_currentFrame;
              if (g_currentFrame != g_script.begin()) {
                  cout << "Not at Begin\n";
                  --g_currentFrame;
                  g_script.erase(tempFrame);
                  
                  load_key();
              }
              else {
                  cout << "at begin\n";
                  ++g_currentFrame;
                  g_script.erase(tempFrame);
                  
                  load_key();
              }
          }
      }
      else if (g_script.size() == 1) {

          list<vector<RigTForm>> empty;
          g_script = empty;
          g_currentFrame = g_script.begin();

      }

      break;

  case ' ':
    g_spaceDown = true;
    break;
  case 27:
    exit(0);                                  // ESC
  case 'h':
    cout << " ============== H E L P ==============\n\n"
    << "h\t\thelp menu\n"
    << "s\t\tsave screenshot\n"
    << "f\t\tToggle flat shading on/off.\n"
    << "p\t\tUse mouse to pick a part to edit\n"
    << "v\t\tCycle view\n"
    << "drag left mouse to rotate\n" << endl;
    break;
  case 's':
    glFlush();
    writePpmScreenshot(g_windowWidth, g_windowHeight, "out.ppm");
    break;
  case 'f':
    g_activeShader = (g_activeShader + 1) % g_numRegularShaders;
    break;
  case 'v':
  {
    shared_ptr<SgRbtNode> viewers[] = {g_skyNode, g_robot1Node, g_robot2Node};
    for (int i = 0; i < 3; ++i) {
      if (g_currentCameraNode == viewers[i]) {
        g_currentCameraNode = viewers[(i+1)%3];
        break;
      }
    }
  }
  break;
  case 'p':
    g_pickingMode = !g_pickingMode;
    cerr << "Picking mode is " << (g_pickingMode ? "on" : "off") << endl;
    break;
  case 'm':
    g_activeCameraFrame = SkyMode((g_activeCameraFrame+1) % 2);
    cerr << "Editing sky eye w.r.t. " << (g_activeCameraFrame == WORLD_SKY ? "world-sky frame\n" : "sky-sky frame\n") << endl;
    break;
  case 'y':
      g_loopState = false;
      if (g_script.size() > 3) {
          animateTimerCallback(0);
      }
      break;
  case 'l':
      g_loopState = true;
      if (g_script.size() > 3) {
          animateTimerCallback(0);
      }
      break;

  case '<':
      move_left();

      break;
  case '>':
      move_right();

      break;
  case 'c':
      if (g_script.size() > 1) {
          load_key();
      }
      else if (firstLeft && g_script.size() > 0) {
          --g_currentFrame;
          firstLeft = false;
          load_key();
      }
      break;
  case 'u':
      if (g_script.size() > 0) {
          save_key();
      }

      break;
  case 'n':
    vector<RigTForm> nextFrame;
    for (vector<shared_ptr<SgRbtNode>>::iterator it = g_map.begin(); it != g_map.end(); ++it) {
        nextFrame.push_back((**it).getRbt());
    }
    if (g_currentFrame != g_script.end()) {
        ++g_currentFrame;
        firstLeft = true;
    }
    g_script.insert(g_currentFrame, nextFrame);
    move_left();
    move_right();
    //--g_currentFrame;
    //cout << distance(g_script.begin(), g_currentFrame);
    //cout << "\n";
    //cout << g_script.size();
    //cout << "\n";
    //print_pos();

    break;
  }
  glutPostRedisplay();
}

static void initGlutState(int argc, char * argv[]) {
  glutInit(&argc, argv);                                  // initialize Glut based on cmd-line args
#ifdef __MAC__
  glutInitDisplayMode(GLUT_3_2_CORE_PROFILE|GLUT_RGBA|GLUT_DOUBLE|GLUT_DEPTH); // core profile flag is required for GL 3.2 on Mac
#else
  glutInitDisplayMode(GLUT_RGBA|GLUT_DOUBLE|GLUT_DEPTH);  //  RGBA pixel channels and double buffering
#endif
  glutInitWindowSize(g_windowWidth, g_windowHeight);      // create a window
  glutCreateWindow("Assignment 6");                       // title the window

  glutIgnoreKeyRepeat(true);                              // avoids repeated keyboard calls when holding space to emulate middle mouse

  glutDisplayFunc(display);                               // display rendering callback
  glutReshapeFunc(reshape);                               // window reshape callback
  glutMotionFunc(motion);                                 // mouse movement callback
  glutMouseFunc(mouse);                                   // mouse click callback
  glutKeyboardFunc(keyboard);
  glutKeyboardUpFunc(keyboardUp);
}

static void initGLState() {
  glClearColor(128./255., 200./255., 255./255., 0.);
  glClearDepth(0.);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glCullFace(GL_BACK);
  glEnable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_GREATER);
  glReadBuffer(GL_BACK);
  if (!g_Gl2Compatible)
    glEnable(GL_FRAMEBUFFER_SRGB);
}

static void initShaders() {
  g_shaderStates.resize(g_numShaders);
  for (int i = 0; i < g_numShaders; ++i) {
    if (g_Gl2Compatible)
      g_shaderStates[i].reset(new ShaderState(g_shaderFilesGl2[i][0], g_shaderFilesGl2[i][1]));
    else
      g_shaderStates[i].reset(new ShaderState(g_shaderFiles[i][0], g_shaderFiles[i][1]));
  }
}

static void initGeometry() {
  initGround();
  initCubes();
  initSphere();
  initRobots();
}

static void constructRobot(shared_ptr<SgTransformNode> base, const Cvec3& color) {
  const float ARM_LEN = 0.7,
               ARM_THICK = 0.25,
               LEG_LEN = 1,
               LEG_THICK = 0.25,
               TORSO_LEN = 1.5,
               TORSO_THICK = 0.25,
               TORSO_WIDTH = 1,
               HEAD_SIZE = 0.7;
  const int NUM_JOINTS = 10,
            NUM_SHAPES = 10;

  struct JointDesc {
    int parent;
    float x, y, z;
  };

  JointDesc jointDesc[NUM_JOINTS] = {
    {-1}, // torso
    {0,  TORSO_WIDTH/2, TORSO_LEN/2, 0}, // upper right arm
    {0, -TORSO_WIDTH/2, TORSO_LEN/2, 0}, // upper left arm
    {1,  ARM_LEN, 0, 0}, // lower right arm
    {2, -ARM_LEN, 0, 0}, // lower left arm
    {0, TORSO_WIDTH/2-LEG_THICK/2, -TORSO_LEN/2, 0}, // upper right leg
    {0, -TORSO_WIDTH/2+LEG_THICK/2, -TORSO_LEN/2, 0}, // upper left leg
    {5, 0, -LEG_LEN, 0}, // lower right leg
    {6, 0, -LEG_LEN, 0}, // lower left
    {0, 0, TORSO_LEN/2, 0} // head
  };

  struct ShapeDesc {
    int parentJointId;
    float x, y, z, sx, sy, sz;
    shared_ptr<Geometry> geometry;
  };

  ShapeDesc shapeDesc[NUM_SHAPES] = {
    {0, 0,         0, 0, TORSO_WIDTH, TORSO_LEN, TORSO_THICK, g_cube}, // torso
    {1, ARM_LEN/2, 0, 0, ARM_LEN/2, ARM_THICK/2, ARM_THICK/2, g_sphere}, // upper right arm
    {2, -ARM_LEN/2, 0, 0, ARM_LEN/2, ARM_THICK/2, ARM_THICK/2, g_sphere}, // upper left arm
    {3, ARM_LEN/2, 0, 0, ARM_LEN, ARM_THICK, ARM_THICK, g_cube}, // lower right arm
    {4, -ARM_LEN/2, 0, 0, ARM_LEN, ARM_THICK, ARM_THICK, g_cube}, // lower left arm
    {5, 0, -LEG_LEN/2, 0, LEG_THICK/2, LEG_LEN/2, LEG_THICK/2, g_sphere}, // upper right leg
    {6, 0, -LEG_LEN/2, 0, LEG_THICK/2, LEG_LEN/2, LEG_THICK/2, g_sphere}, // upper left leg
    {7, 0, -LEG_LEN/2, 0, LEG_THICK, LEG_LEN, LEG_THICK, g_cube}, // lower right leg
    {8, 0, -LEG_LEN/2, 0, LEG_THICK, LEG_LEN, LEG_THICK, g_cube}, // lower left leg
    {9, 0, static_cast<float>(HEAD_SIZE/2 * 1.5), 0, HEAD_SIZE/2, HEAD_SIZE/2, HEAD_SIZE/2, g_sphere}, // head
  };

  shared_ptr<SgTransformNode> jointNodes[NUM_JOINTS];

  for (int i = 0; i < NUM_JOINTS; ++i) {
    if (jointDesc[i].parent == -1)
      jointNodes[i] = base;
    else {
      jointNodes[i].reset(new SgRbtNode(RigTForm(Cvec3(jointDesc[i].x, jointDesc[i].y, jointDesc[i].z))));
      jointNodes[jointDesc[i].parent]->addChild(jointNodes[i]);
    }
  }
  for (int i = 0; i < NUM_SHAPES; ++i) {
    shared_ptr<MyShapeNode> shape(
      new MyShapeNode(shapeDesc[i].geometry,
                      color,
                      Cvec3(shapeDesc[i].x, shapeDesc[i].y, shapeDesc[i].z),
                      Cvec3(0, 0, 0),
                      Cvec3(shapeDesc[i].sx, shapeDesc[i].sy, shapeDesc[i].sz)));
    jointNodes[shapeDesc[i].parentJointId]->addChild(shape);
  }
}

static void initScene() {


  g_world.reset(new SgRootNode());

  g_skyNode.reset(new SgRbtNode(RigTForm(Cvec3(0.0, 0.25, 4.0))));

  g_groundNode.reset(new SgRbtNode());
  g_groundNode->addChild(shared_ptr<MyShapeNode>(
                           new MyShapeNode(g_ground, Cvec3(0.1, 0.95, 0.1))));

  g_robot1Node.reset(new SgObjectNode(RigTForm(Cvec3(-2, 1, 0)), Cvec3(.1, 0, 0), Cvec3(0, 0, 0)));
  g_robot2Node.reset(new SgRbtNode(RigTForm(Cvec3(2, 2, 0))));

  g_objectNodes.push_back(g_robot1Node);
  constructRobot(g_robot1Node, Cvec3(1, 0, 0)); // a Red robot
  constructRobot(g_robot2Node, Cvec3(0, 0, 1)); // a Blue robot

  g_world->addChild(g_skyNode);
  g_world->addChild(g_groundNode);
  g_world->addChild(g_robot1Node);
  g_world->addChild(g_robot2Node);

  g_currentCameraNode = g_skyNode;
  dumpSgRbtNodes(g_world, g_map);
}

int main(int argc, char * argv[]) {
  try {
    initGlutState(argc,argv);

    // on Mac, we shouldn't use GLEW.

#ifndef __MAC__
    glewInit(); // load the OpenGL extensions
#endif

    cout << (g_Gl2Compatible ? "Will use OpenGL 2.x / GLSL 1.0" : "Will use OpenGL 3.x / GLSL 1.5") << endl;

#ifndef __MAC__
    if ((!g_Gl2Compatible) && !GLEW_VERSION_3_0)
      throw runtime_error("Error: card/driver does not support OpenGL Shading Language v1.3");
    else if (g_Gl2Compatible && !GLEW_VERSION_2_0)
      throw runtime_error("Error: card/driver does not support OpenGL Shading Language v1.0");
#endif

    initGLState();
    initShaders();
    initGeometry();
    initScene();

    glutMainLoop();
    return 0;
  }
  catch (const runtime_error& e) {
    cout << "Exception caught: " << e.what() << endl;
    return -1;
  }
}
