#ifndef PBRT_CORE_LIGHTTREE_H
#define PBRT_CORE_LIGHTTREE_H

#include "lightdistrib.h"

namespace pbrt {

class SGLightTree;

class SGLightTreeDistribution : public LightDistribution {
  public:
    SGLightTreeDistribution(const Scene &scene);
    ~SGLightTreeDistribution();

    const Distribution1D *Lookup(const Point3f &p) const;
    bool SampleLight(const Interaction &it, Float u, int *lightNum,
                     Float *pdf) const;

  private:
    std::unique_ptr<SGLightTree> tree;
    std::unique_ptr<LightDistribution> fallback;
};

}  // namespace pbrt

#endif  // PBRT_CORE_LIGHTTREE_H
