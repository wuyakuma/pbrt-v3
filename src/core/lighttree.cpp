#include "lighttree.h"

#include "interaction.h"
#include "lights/diffuse.h"
#include "reflection.h"
#include "scene.h"
#include "sg.h"
#include "shapes/triangle.h"

#include <algorithm>

namespace pbrt {

namespace {

struct LightTreeCluster {
    Float flux = 0;
    Point3f mean;
    Float spatialVariance = 0;
    Float radius = 0;
    Vector3f averageDirection = Vector3f(0, 0, 0);
    Vector3f vmfAxis = Vector3f(0, 0, 1);
    Float vmfSharpness = 0;
    Bounds3f bounds;
};

struct LightTreeNode {
    int left = -1, right = -1;
    int lightIndex = -1;
    LightTreeCluster cluster;
    bool IsLeaf() const { return lightIndex >= 0; }
};

struct ViewedSGLight {
    SGLobe lobe;
    Float amplitude = 0;
};

static Float SpectrumY(const Spectrum &s) { return std::max(Float(0), s.y()); }

static LightTreeCluster MakeGenericCluster(const Scene &scene, int lightIndex) {
    const std::shared_ptr<Light> &light = scene.lights[lightIndex];
    LightTreeCluster c;
    c.flux = SpectrumY(light->Power());

    Point3f center;
    Float sceneRadius;
    scene.WorldBound().BoundingSphere(&center, &sceneRadius);
    Interaction ref(center, Normal3f(), Vector3f(), Vector3f(1, 0, 0), 0,
                    MediumInterface());
    Float pdf = 0;
    Vector3f wi(0, 0, 1);
    VisibilityTester vis;
    Spectrum Li = light->Sample_Li(ref, Point2f(Float(0.5), Float(0.5)), &wi,
                                   &pdf, &vis);
    if (pdf > 0 && !Li.IsBlack()) {
        c.mean = vis.P1().p;
        c.averageDirection = IsDeltaLight(light->flags) ? Vector3f(0, 0, 0) : -wi;
    } else {
        c.mean = center;
        c.averageDirection = Vector3f(0, 0, 0);
    }

    c.bounds = Bounds3f(c.mean);
    c.radius = 0;
    c.spatialVariance = 0;
    Float len = c.averageDirection.Length();
    if (len > 0) {
        c.vmfAxis = c.averageDirection / len;
        c.vmfSharpness = VMFAxisLengthToSharpness(len);
    }
    return c;
}

static LightTreeCluster MakeDiffuseTriangleCluster(const DiffuseAreaLight *light,
                                                   const Triangle *triangle,
                                                   int lightIndex) {
    LightTreeCluster c;
    c.flux = SpectrumY(light->Power());
    Point3f p[3];
    triangle->GetVertices(p);
    Vector3f e1 = p[1] - p[0], e2 = p[2] - p[0];
    c.mean = p[0] + (e1 + e2) / 3;
    c.spatialVariance =
        (e1.LengthSquared() + e2.LengthSquared() - Dot(e1, e2)) / 18;
    c.bounds = triangle->WorldBound();
    c.radius = std::sqrt(std::max({DistanceSquared(c.mean, p[0]),
                                   DistanceSquared(c.mean, p[1]),
                                   DistanceSquared(c.mean, p[2])}));
    Vector3f n = Normalize(Cross(e1, e2));
    c.averageDirection = light->IsTwoSided() ? Vector3f(0, 0, 0) : Float(0.5) * n;
    Float len = c.averageDirection.Length();
    if (len > 0) {
        c.vmfAxis = c.averageDirection / len;
        c.vmfSharpness = VMFAxisLengthToSharpness(len);
    }
    return c;
}

static LightTreeCluster MakeClusterForLight(const Scene &scene, int lightIndex) {
    const DiffuseAreaLight *diffuse =
        dynamic_cast<const DiffuseAreaLight *>(scene.lights[lightIndex].get());
    if (diffuse) {
        const Triangle *triangle =
            dynamic_cast<const Triangle *>(diffuse->GetShape().get());
        if (triangle) return MakeDiffuseTriangleCluster(diffuse, triangle, lightIndex);
    }
    return MakeGenericCluster(scene, lightIndex);
}

static LightTreeCluster CombineClusters(const LightTreeCluster &a,
                                        const LightTreeCluster &b) {
    LightTreeCluster c;
    c.flux = a.flux + b.flux;
    if (c.flux <= 0) {
        c.mean = Point3f((a.mean.x + b.mean.x) / 2, (a.mean.y + b.mean.y) / 2,
                         (a.mean.z + b.mean.z) / 2);
        c.bounds = Union(a.bounds, b.bounds);
        return c;
    }

    Float wa = a.flux / c.flux, wb = b.flux / c.flux;
    c.mean = Point3f(wa * a.mean.x + wb * b.mean.x,
                     wa * a.mean.y + wb * b.mean.y,
                     wa * a.mean.z + wb * b.mean.z);
    c.spatialVariance = wa * a.spatialVariance + wb * b.spatialVariance +
                        wa * wb * DistanceSquared(a.mean, b.mean);
    c.bounds = Union(a.bounds, b.bounds);
    c.radius = std::max(Distance(c.mean, a.mean) + a.radius,
                        Distance(c.mean, b.mean) + b.radius);
    c.averageDirection = wa * a.averageDirection + wb * b.averageDirection;
    Float len = c.averageDirection.Length();
    if (len > 0) {
        c.vmfAxis = c.averageDirection / len;
        c.vmfSharpness = VMFAxisLengthToSharpness(len);
    }
    return c;
}

static bool ViewedLight(const LightTreeCluster &cluster, const Interaction &it,
                        ViewedSGLight *viewed) {
    Vector3f lightVector = cluster.mean - it.p;
    Float dist2 = lightVector.LengthSquared();
    if (dist2 <= 0 || cluster.flux <= 0) return false;
    Float dist = std::sqrt(dist2);
    Vector3f lightDir = lightVector / dist;

    Float c = 0;
    if (it.IsSurfaceInteraction()) {
        const SurfaceInteraction &si = (const SurfaceInteraction &)it;
        c = std::max(Float(0), Dot(Vector3f(si.shading.n), Normalize(it.p - cluster.mean)));
    }
    Float variance = Lerp(c, cluster.spatialVariance, Float(0.5) * cluster.radius * cluster.radius);
    const Float maxSharpness = Float(1e6);
    variance = std::max(variance, dist2 / maxSharpness);
    variance = std::max(variance, Float(MachineEpsilon));

    Float spatialSharpness = dist2 / variance;
    viewed->lobe =
        SGProduct(-cluster.vmfAxis, cluster.vmfSharpness, lightDir, spatialSharpness);
    Float denom = 2 * Pi * variance * SGIntegral(cluster.vmfSharpness);
    viewed->amplitude =
        cluster.flux * std::exp(viewed->lobe.logAmplitude) /
        std::max(denom, Float(MachineEpsilon));
    return viewed->amplitude > 0 && viewed->lobe.sharpness >= 0;
}

static Float DiffuseImportance(const ViewedSGLight &viewed,
                               const SurfaceInteraction &si,
                               const Spectrum &diffuseWeight) {
    Float diffuse = SpectrumY(diffuseWeight);
    if (diffuse == 0) return 0;
    Float cosine = Clamp(Dot(viewed.lobe.axis, Vector3f(si.shading.n)), -1, 1);
    return diffuse * viewed.amplitude *
           SGClampedCosineProductIntegralOverPi2024(cosine, viewed.lobe.sharpness);
}

static Float GlossyImportance(const ViewedSGLight &viewed,
                              const SurfaceInteraction &si,
                              const Spectrum &glossyWeight, Float alphax,
                              Float alphay) {
    Float glossy = SpectrumY(glossyWeight);
    if (glossy == 0 || !si.bsdf) return 0;

    Vector3f wo = si.bsdf->WorldToLocal(si.wo);
    if (std::abs(wo.z) <= Float(MachineEpsilon)) return 0;
    Vector3f axis = si.bsdf->WorldToLocal(viewed.lobe.axis);
    Float lightLobeVariance =
        1 / std::max(viewed.lobe.sharpness, Float(MachineEpsilon));

    Float ax2 = alphax * alphax, ay2 = alphay * alphay;
    Float projx = ax2 / std::max(Float(1) - ax2, Float(MachineEpsilon));
    Float projy = ay2 / std::max(Float(1) - ay2, Float(MachineEpsilon));

    Float vlen = std::sqrt(wo.x * wo.x + wo.y * wo.y);
    Float vx = vlen > 0 ? wo.x / vlen : 1;
    Float vy = vlen > 0 ? wo.y / vlen : 0;
    Float j0 = Float(0.25);
    Float j1 = Float(0.25) / std::max(wo.z * wo.z, Float(MachineEpsilon));
    SGMatrix2x2 jj(vx * vx * j0 + vy * vy * j1,
                   vx * vy * (j0 - j1),
                   vx * vy * (j0 - j1),
                   vy * vy * j0 + vx * vx * j1);

    SGMatrix2x2 filteredProj(
        projx + 2 * lightLobeVariance * jj.m00,
        2 * lightLobeVariance * jj.m01,
        2 * lightLobeVariance * jj.m10,
        projy + 2 * lightLobeVariance * jj.m11);
    Float det =
        std::max(SGMatrixDeterminant(filteredProj), Float(MachineEpsilon));
    Float tr = filteredProj.m00 + filteredProj.m11;
    Float denom = std::max(Float(1) + tr + det, Float(MachineEpsilon));
    SGMatrix2x2 filteredRoughness((filteredProj.m00 + det) / denom,
                                  filteredProj.m01 / denom,
                                  filteredProj.m10 / denom,
                                  (filteredProj.m11 + det) / denom);

    Vector3f halfVector = wo + axis;
    if (halfVector.LengthSquared() <= Float(MachineEpsilon)) return 0;
    halfVector = Normalize(halfVector);
    Float lobe = SGGXReflectionPDF(wo, halfVector, filteredRoughness);

    Float roughnessMax2 = std::max(ax2, ay2);
    Float reflecSharpness =
        (1 - roughnessMax2) /
        std::max(2 * roughnessMax2, Float(MachineEpsilon));
    Vector3f dominantNormal = GGXDominantVisibleNormal(wo, alphax, alphay);
    Vector3f reflecVec = Reflect(-wo, dominantNormal) * reflecSharpness;
    Vector3f prodVec = reflecVec + axis * viewed.lobe.sharpness;
    Float prodSharpness = prodVec.Length();
    Float visibility = prodSharpness > 0
                           ? VMFHemisphericalIntegral((prodVec / prodSharpness).z,
                                                       prodSharpness)
                           : Float(0.5);
    return glossy * viewed.amplitude * visibility * lobe *
           SGIntegral(viewed.lobe.sharpness);
}

}  // namespace

class SGLightTree {
  public:
    SGLightTree(const Scene &scene) {
        nodes.reserve(scene.lights.size() * 2);
        primitives.reserve(scene.lights.size());
        for (int i = 0; i < (int)scene.lights.size(); ++i)
            primitives.push_back(MakeClusterForLight(scene, i));
        std::vector<int> indices(scene.lights.size());
        for (int i = 0; i < (int)indices.size(); ++i) indices[i] = i;
        if (!indices.empty()) root = Build(indices, 0, indices.size());
    }

    bool Empty() const { return root < 0; }

    bool Sample(const Interaction &it, Float u, int *lightNum, Float *pdf) const {
        if (root < 0) return false;
        u = std::min(u, OneMinusEpsilon);
        int nodeIndex = root;
        *pdf = 1;
        while (!nodes[nodeIndex].IsLeaf()) {
            const LightTreeNode &node = nodes[nodeIndex];
            Float leftImportance = Importance(nodes[node.left].cluster, it);
            Float rightImportance = Importance(nodes[node.right].cluster, it);
            Float sum = leftImportance + rightImportance;
            Float pLeft = sum > 0 ? leftImportance / sum : Float(0.5);
            pLeft = Clamp(pLeft, Float(0), Float(1));
            if (u < pLeft) {
                *pdf *= pLeft;
                u = pLeft > 0 ? u / pLeft : 0;
                nodeIndex = node.left;
            } else {
                Float pRight = 1 - pLeft;
                *pdf *= pRight;
                u = pRight > 0 ? (u - pLeft) / pRight : 0;
                nodeIndex = node.right;
            }
            if (*pdf <= 0) return false;
        }
        *lightNum = nodes[nodeIndex].lightIndex;
        return *pdf > 0;
    }

  private:
    int Build(std::vector<int> &indices, size_t start, size_t end) {
        int nodeIndex = nodes.size();
        nodes.push_back(LightTreeNode());
        if (end - start == 1) {
            int primIndex = indices[start];
            nodes[nodeIndex].lightIndex = primIndex;
            nodes[nodeIndex].cluster = primitives[primIndex];
            return nodeIndex;
        }

        Bounds3f centroidBounds;
        for (size_t i = start; i < end; ++i)
            centroidBounds = Union(centroidBounds, primitives[indices[i]].mean);
        int axis = centroidBounds.MaximumExtent();
        size_t mid = (start + end) / 2;
        std::nth_element(indices.begin() + start, indices.begin() + mid,
                         indices.begin() + end, [&](int a, int b) {
                             return primitives[a].mean[axis] < primitives[b].mean[axis];
                         });
        nodes[nodeIndex].left = Build(indices, start, mid);
        nodes[nodeIndex].right = Build(indices, mid, end);
        nodes[nodeIndex].cluster =
            CombineClusters(nodes[nodes[nodeIndex].left].cluster,
                            nodes[nodes[nodeIndex].right].cluster);
        return nodeIndex;
    }

    Float Importance(const LightTreeCluster &cluster, const Interaction &it) const {
        ViewedSGLight viewed;
        if (!ViewedLight(cluster, it, &viewed)) return 0;

        Float importance = 0;
        if (it.IsSurfaceInteraction()) {
            const SurfaceInteraction &si = (const SurfaceInteraction &)it;
            if (si.bsdf) {
                Spectrum diffuseWeight, glossyWeight;
                Float alphax, alphay;
                si.bsdf->SGSamplingInfo(si.wo, &diffuseWeight, &glossyWeight,
                                        &alphax, &alphay);
                importance += DiffuseImportance(viewed, si, diffuseWeight);
                importance +=
                    GlossyImportance(viewed, si, glossyWeight, alphax, alphay);
            }
        } else {
            importance = viewed.amplitude * SGIntegral(viewed.lobe.sharpness) * Inv4Pi;
        }

        Float distance2 =
            std::max(DistanceSquared(it.p, cluster.mean), Float(MachineEpsilon));
        Float floor = Float(1e-6) * cluster.flux / distance2;
        return std::max(importance, floor);
    }

    std::vector<LightTreeCluster> primitives;
    std::vector<LightTreeNode> nodes;
    int root = -1;
};

SGLightTreeDistribution::SGLightTreeDistribution(const Scene &scene)
    : tree(new SGLightTree(scene)), fallback(new PowerLightDistribution(scene)) {}

SGLightTreeDistribution::~SGLightTreeDistribution() {}

const Distribution1D *SGLightTreeDistribution::Lookup(const Point3f &p) const {
    return fallback->Lookup(p);
}

bool SGLightTreeDistribution::SampleLight(const Interaction &it, Float u,
                                          int *lightNum, Float *pdf) const {
    if (tree && !tree->Empty() && tree->Sample(it, u, lightNum, pdf)) return true;
    return fallback->SampleLight(it, u, lightNum, pdf);
}

}  // namespace pbrt
