#ifndef SCENEGRAPH_H
#define SCENEGRAPH_H

#include <vector>
#include <memory>
#include <stdexcept>

#include "cvec.h"
#include "matrix4.h"
#include "rigtform.h"
#include "glsupport.h" // for Noncopyable
#include "asstcommon.h"

class SgNodeVisitor;

class SgNode : public std::enable_shared_from_this<SgNode>, Noncopyable {
public:
  virtual bool accept(SgNodeVisitor& vistor) = 0;
  virtual ~SgNode() {}

  // Two nodes are equal if and only if they're the same, i.e.,
  // having the same in memory address
  bool operator == (const SgNode& other) const {
    return this == &other;
  }

  bool operator != (const SgNode& other) const {
    return !(*this == other);
  }

protected:
  SgNode() {}
};

//
// A transform node can have descendents nodes. It uses a
// rigid body transform to represent its frame with respect to
// the parent frame
//
class SgTransformNode : public SgNode {
public:
  virtual bool accept(SgNodeVisitor& visitor);
  virtual RigTForm getRbt() = 0;

  void addChild(std::shared_ptr<SgNode> child);
  void removeChild(std::shared_ptr<SgNode> child);

  int getNumChildren() const {
    return children_.size();
  }

  std::shared_ptr<SgNode> getChild(int i) {
    return children_[i];
  }

private:
  std::vector<std::shared_ptr<SgNode>> children_;
};


//
// A shape node has no decendents, but can perform drawing. It can also
// provide an affine matrix to be used to modify its geometry.
//
class SgShapeNode : public SgNode {
public:
  virtual bool accept(SgNodeVisitor& visitor);

  virtual Matrix4 getAffineMatrix() = 0;
  virtual void draw(const ShaderState& curSS) = 0;
};


// Visitor class for the scene graph nodes. If any of the
// visit/postVisit functions return false, the traverse
// will be terminated.
class SgNodeVisitor {
public:
  virtual bool visit(SgTransformNode& node) { return true; }
  virtual bool visit(SgShapeNode& node) { return true; }

  virtual bool postVisit(SgTransformNode& node) { return true; }
  virtual bool postVisit(SgShapeNode& node) { return true; }
};


RigTForm getPathAccumRbt(
  std::shared_ptr<SgTransformNode> source,
  std::shared_ptr<SgTransformNode> destination,
  int offsetFromDestination = 0);


//----------------------------------------------------
// Concrete scene graph node implementations follow
//----------------------------------------------------

// A SgRoot node is a Transform node with identity Rbt
class SgRootNode : public SgTransformNode {
public:
  SgRootNode() {}

  virtual RigTForm getRbt() {
    return RigTForm();
  }
};

// A SgRbtNode is a Transform node that wraps a RigTForm
class SgRbtNode : public SgTransformNode {
public:
  SgRbtNode(const RigTForm& rbt = RigTForm())
    : rbt_ (rbt) {}

  virtual RigTForm getRbt() {
    return rbt_;
  }

  void setRbt(const RigTForm& rbt) {
    rbt_ = rbt;
  }

private:
  RigTForm rbt_;
};

//Final Project section
class SgObjectNode : public SgRbtNode {
public:
    SgObjectNode(const RigTForm& rbt = RigTForm(), const Cvec3 velo = Cvec3(), const Cvec3 velo_buffer = Cvec3())
        : _velo(velo) {
        setRbt(rbt);
    }

    void accelerate(Cvec3 acceleration) {
        _velo_buffer = _velo + acceleration;
    }
    Cvec3 getVelocity() {
        return _velo_buffer;
    }
    void setVelocity(Cvec3 newVelo) {
        _velo_buffer = newVelo;
    }
    void update() {
        setRbt(RigTForm(getRbt().getTranslation() +getRbt().getRotation()*_velo, getRbt().getRotation()));
        _velo = _velo_buffer;
    }
    void setMass(double mass) {
        _mass = mass;
    }
    double getMass() {
        return _mass;
    }
private:
    Cvec3 _velo;
    Cvec3 _velo_buffer = _velo;
    double _mass = 1;
};

// A SgGeometryShapeNode is a Shape node that wraps a user geometry class
template<typename Geometry>
class SgGeometryShapeNode : public SgShapeNode {
  std::shared_ptr<Geometry> geometry_;
  Matrix4 affineMatrix_;
  Cvec3 color_;
public:
  SgGeometryShapeNode(std::shared_ptr<Geometry> geometry,
                      const Cvec3& color,
                      const Cvec3& translation = Cvec3(0, 0, 0),
                      const Cvec3& eulerAngles = Cvec3(0, 0, 0),
                      const Cvec3& scales = Cvec3(1, 1, 1))
    : geometry_(geometry)
    , color_(color)
    , affineMatrix_(Matrix4::makeTranslation(translation) *
                    Matrix4::makeXRotation(eulerAngles[0]) *
                    Matrix4::makeYRotation(eulerAngles[1]) *
                    Matrix4::makeZRotation(eulerAngles[2]) *
                    Matrix4::makeScale(scales)) {}

  virtual Matrix4 getAffineMatrix() {
    return affineMatrix_;
  }

  virtual void draw(const ShaderState& curSS) {
    safe_glUniform3f(curSS.h_uColor, color_[0], color_[1], color_[2]);
    geometry_->draw(curSS);
  }
};

#endif
