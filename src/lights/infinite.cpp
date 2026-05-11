
/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

// lights/infinite.cpp*
#include "lights/infinite.h"
#include "imageio.h"
#include "paramset.h"
#include "reflection.h"
#include "sampling.h"
#include "sg.h"
#include "stats.h"

namespace pbrt {

namespace {

struct SGEnvNode {
    int left = -1, right = -1;
    int leaf = -1;
    int tx0 = 0, tx1 = 0, ty0 = 0, ty1 = 0;
    Float flux = 0;
    Float solidAngle = 0;
    Float sharpness = 0;
    Vector3f axis = Vector3f(0, 0, 1);
    Vector3f moment = Vector3f(0, 0, 0);
    Float momentWeight = 0;
    bool IsLeaf() const { return leaf >= 0; }
};

struct SGEnvLeaf {
    int tx = 0, ty = 0;
    int u0 = 0, u1 = 0, v0 = 0, v1 = 0;
    Float phi0 = 0, phi1 = 0;
    Float cosTheta0 = 1, cosTheta1 = -1;
    Float solidAngle = 0;
    Float flux = 0;
    Vector3f axis = Vector3f(0, 0, 1);
};

static Float SpectrumY(const Spectrum &s) { return std::max(Float(0), s.y()); }

static int CeilDiv(int a, int b) { return (a + b - 1) / b; }

static Vector3f LatLongDirection(Float phi, Float theta) {
    Float sinTheta = std::sin(theta);
    return Vector3f(sinTheta * std::cos(phi), sinTheta * std::sin(phi),
                    std::cos(theta));
}

static Float SphericalRectSolidAngle(Float phi0, Float phi1, Float theta0,
                                     Float theta1) {
    return (phi1 - phi0) * std::max(Float(0), std::cos(theta0) - std::cos(theta1));
}

static void FinalizeSGNode(SGEnvNode *node) {
    Float len = node->moment.Length();
    if (len > 0) {
        node->axis = node->moment / len;
        Float axisLength = len / std::max(node->momentWeight, Float(MachineEpsilon));
        node->sharpness =
            std::min(VMFAxisLengthToSharpness(axisLength), Float(1e6));
    }
}

static Float SGDiffuseImportance(const Vector3f &axis, Float sharpness,
                                 Float amplitude,
                                 const SurfaceInteraction &si,
                                 const Spectrum &diffuseWeight) {
    Float diffuse = SpectrumY(diffuseWeight);
    if (diffuse == 0) return 0;
    Float cosine = Clamp(Dot(axis, Vector3f(si.shading.n)), -1, 1);
    return diffuse * amplitude *
           SGClampedCosineProductIntegralOverPi2024(cosine, sharpness);
}

static Float SGGlossyImportance(const Vector3f &axis, Float sharpness,
                                Float amplitude,
                                const SurfaceInteraction &si,
                                const Spectrum &glossyWeight, Float alphax,
                                Float alphay) {
    Float glossy = SpectrumY(glossyWeight);
    if (glossy == 0 || !si.bsdf) return 0;

    Vector3f wo = si.bsdf->WorldToLocal(si.wo);
    if (std::abs(wo.z) <= Float(MachineEpsilon)) return 0;
    Vector3f localAxis = si.bsdf->WorldToLocal(axis);
    Float lightLobeVariance = 1 / std::max(sharpness, Float(MachineEpsilon));

    Float ax2 = alphax * alphax, ay2 = alphay * alphay;
    Float projx = ax2 / std::max(Float(1) - ax2, Float(MachineEpsilon));
    Float projy = ay2 / std::max(Float(1) - ay2, Float(MachineEpsilon));

    Float vlen = std::sqrt(wo.x * wo.x + wo.y * wo.y);
    Float vx = vlen > 0 ? wo.x / vlen : 1;
    Float vy = vlen > 0 ? wo.y / vlen : 0;
    Float j0 = Float(0.25);
    Float j1 = Float(0.25) / std::max(wo.z * wo.z, Float(MachineEpsilon));
    SGMatrix2x2 jj(vx * vx * j0 + vy * vy * j1, vx * vy * (j0 - j1),
                   vx * vy * (j0 - j1), vy * vy * j0 + vx * vx * j1);

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

    Vector3f halfVector = wo + localAxis;
    if (halfVector.LengthSquared() <= Float(MachineEpsilon)) return 0;
    halfVector = Normalize(halfVector);
    Float lobe = SGGXReflectionPDF(wo, halfVector, filteredRoughness);

    Float roughnessMax2 = std::max(ax2, ay2);
    Float reflecSharpness =
        (1 - roughnessMax2) /
        std::max(2 * roughnessMax2, Float(MachineEpsilon));
    Vector3f dominantNormal = GGXDominantVisibleNormal(wo, alphax, alphay);
    Vector3f reflecVec = Reflect(-wo, dominantNormal) * reflecSharpness;
    Vector3f prodVec = reflecVec + localAxis * sharpness;
    Float prodSharpness = prodVec.Length();
    Float visibility = prodSharpness > 0
                           ? VMFHemisphericalIntegral((prodVec / prodSharpness).z,
                                                       prodSharpness)
                           : Float(0.5);
    return glossy * amplitude * visibility * lobe * SGIntegral(sharpness);
}

}  // namespace

// InfiniteAreaLight Method Definitions
InfiniteAreaLight::InfiniteAreaLight(const Transform &LightToWorld,
                                     const Spectrum &L, int nSamples,
                                     const std::string &texmap)
    : Light((int)LightFlags::Infinite, LightToWorld, MediumInterface(),
            nSamples) {
    // Read texel data from _texmap_ and initialize _Lmap_
    Point2i resolution;
    std::unique_ptr<RGBSpectrum[]> texels(nullptr);
    if (texmap != "") {
        texels = ReadImage(texmap, &resolution);
        if (texels)
            for (int i = 0; i < resolution.x * resolution.y; ++i)
                texels[i] *= L.ToRGBSpectrum();
    }
    if (!texels) {
        resolution.x = resolution.y = 1;
        texels = std::unique_ptr<RGBSpectrum[]>(new RGBSpectrum[1]);
        texels[0] = L.ToRGBSpectrum();
    }
    Lmap.reset(new MIPMap<RGBSpectrum>(resolution, texels.get()));

    // Initialize sampling PDFs for infinite area light

    // Compute scalar-valued image _img_ from environment map
    int width = 2 * Lmap->Width(), height = 2 * Lmap->Height();
    std::unique_ptr<Float[]> img(new Float[width * height]);
    float fwidth = 0.5f / std::min(width, height);
    ParallelFor(
        [&](int64_t v) {
            Float vp = (v + .5f) / (Float)height;
            Float sinTheta = std::sin(Pi * (v + .5f) / height);
            for (int u = 0; u < width; ++u) {
                Float up = (u + .5f) / (Float)width;
                img[u + v * width] = Lmap->Lookup(Point2f(up, vp), fwidth).y();
                img[u + v * width] *= sinTheta;
            }
        },
        height, 32);

    // Compute sampling distributions for rows and columns of image
    distribution.reset(new Distribution2D(img.get(), width, height));
}

Spectrum InfiniteAreaLight::Power() const {
    return Pi * worldRadius * worldRadius *
           Spectrum(Lmap->Lookup(Point2f(.5f, .5f), .5f),
                    SpectrumType::Illuminant);
}

Spectrum InfiniteAreaLight::Le(const RayDifferential &ray) const {
    Vector3f w = Normalize(WorldToLight(ray.d));
    Point2f st(SphericalPhi(w) * Inv2Pi, SphericalTheta(w) * InvPi);
    return Spectrum(Lmap->Lookup(st), SpectrumType::Illuminant);
}

Spectrum InfiniteAreaLight::Sample_Li(const Interaction &ref, const Point2f &u,
                                      Vector3f *wi, Float *pdf,
                                      VisibilityTester *vis) const {
    ProfilePhase _(Prof::LightSample);
    // Find $(u,v)$ sample coordinates in infinite light texture
    Float mapPdf;
    Point2f uv = distribution->SampleContinuous(u, &mapPdf);
    if (mapPdf == 0) return Spectrum(0.f);

    // Convert infinite light sample point to direction
    Float theta = uv[1] * Pi, phi = uv[0] * 2 * Pi;
    Float cosTheta = std::cos(theta), sinTheta = std::sin(theta);
    Float sinPhi = std::sin(phi), cosPhi = std::cos(phi);
    *wi =
        LightToWorld(Vector3f(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta));

    // Compute PDF for sampled infinite light direction
    *pdf = mapPdf / (2 * Pi * Pi * sinTheta);
    if (sinTheta == 0) *pdf = 0;

    // Return radiance value for infinite light direction
    *vis = VisibilityTester(ref, Interaction(ref.p + *wi * (2 * worldRadius),
                                             ref.time, mediumInterface));
    return Spectrum(Lmap->Lookup(uv), SpectrumType::Illuminant);
}

Float InfiniteAreaLight::Pdf_Li(const Interaction &, const Vector3f &w) const {
    ProfilePhase _(Prof::LightPdf);
    Vector3f wi = WorldToLight(w);
    Float theta = SphericalTheta(wi), phi = SphericalPhi(wi);
    Float sinTheta = std::sin(theta);
    if (sinTheta == 0) return 0;
    return distribution->Pdf(Point2f(phi * Inv2Pi, theta * InvPi)) /
           (2 * Pi * Pi * sinTheta);
}

Spectrum InfiniteAreaLight::Sample_Le(const Point2f &u1, const Point2f &u2,
                                      Float time, Ray *ray, Normal3f *nLight,
                                      Float *pdfPos, Float *pdfDir) const {
    ProfilePhase _(Prof::LightSample);
    // Compute direction for infinite light sample ray
    Point2f u = u1;

    // Find $(u,v)$ sample coordinates in infinite light texture
    Float mapPdf;
    Point2f uv = distribution->SampleContinuous(u, &mapPdf);
    if (mapPdf == 0) return Spectrum(0.f);
    Float theta = uv[1] * Pi, phi = uv[0] * 2.f * Pi;
    Float cosTheta = std::cos(theta), sinTheta = std::sin(theta);
    Float sinPhi = std::sin(phi), cosPhi = std::cos(phi);
    Vector3f d =
        -LightToWorld(Vector3f(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta));
    *nLight = (Normal3f)d;

    // Compute origin for infinite light sample ray
    Vector3f v1, v2;
    CoordinateSystem(-d, &v1, &v2);
    Point2f cd = ConcentricSampleDisk(u2);
    Point3f pDisk = worldCenter + worldRadius * (cd.x * v1 + cd.y * v2);
    *ray = Ray(pDisk + worldRadius * -d, d, Infinity, time);

    // Compute _InfiniteAreaLight_ ray PDFs
    *pdfDir = sinTheta == 0 ? 0 : mapPdf / (2 * Pi * Pi * sinTheta);
    *pdfPos = 1 / (Pi * worldRadius * worldRadius);
    return Spectrum(Lmap->Lookup(uv), SpectrumType::Illuminant);
}

void InfiniteAreaLight::Pdf_Le(const Ray &ray, const Normal3f &, Float *pdfPos,
                               Float *pdfDir) const {
    ProfilePhase _(Prof::LightPdf);
    Vector3f d = -WorldToLight(ray.d);
    Float theta = SphericalTheta(d), phi = SphericalPhi(d);
    Point2f uv(phi * Inv2Pi, theta * InvPi);
    Float mapPdf = distribution->Pdf(uv);
    *pdfDir = mapPdf / (2 * Pi * Pi * std::sin(theta));
    *pdfPos = 1 / (Pi * worldRadius * worldRadius);
}

std::shared_ptr<InfiniteAreaLight> CreateInfiniteLight(
    const Transform &light2world, const ParamSet &paramSet) {
    Spectrum L = paramSet.FindOneSpectrum("L", Spectrum(1.0));
    Spectrum sc = paramSet.FindOneSpectrum("scale", Spectrum(1.0));
    std::string texmap = paramSet.FindOneFilename("mapname", "");
    int nSamples = paramSet.FindOneInt("samples",
                                       paramSet.FindOneInt("nsamples", 1));
    if (PbrtOptions.quickRender) nSamples = std::max(1, nSamples / 4);
    return std::make_shared<InfiniteAreaLight>(light2world, L * sc, nSamples,
                                               texmap);
}

class SGEnvLightTree {
  public:
    SGEnvLightTree(const MIPMap<RGBSpectrum> &lmap, int width, int height,
                   int tileSize)
        : lmap(lmap),
          width(width),
          height(height),
          tileSize(std::max(1, tileSize)),
          tilesX(CeilDiv(width, std::max(1, tileSize))),
          tilesY(CeilDiv(height, std::max(1, tileSize))),
          leafForTile(tilesX * tilesY, -1) {
        if (width > 0 && height > 0) root = Build(0, tilesX, 0, tilesY);
    }

    bool Empty() const { return root < 0 || leaves.empty(); }

    bool Sample(const Interaction *ref, const Transform &LightToWorld, Float u,
                int *leafIndex, Float *pdf, Float *uRemapped) const {
        if (Empty()) return false;
        u = std::min(u, OneMinusEpsilon);
        int nodeIndex = root;
        *pdf = 1;
        while (!nodes[nodeIndex].IsLeaf()) {
            const SGEnvNode &node = nodes[nodeIndex];
            Float pLeft = ChildProbability(node, ref, LightToWorld);
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
        *leafIndex = nodes[nodeIndex].leaf;
        *uRemapped = u;
        return *pdf > 0;
    }

    Point2f SampleLeaf(int leafIndex, Float u0, Float u1,
                       Vector3f *localDir, Float *pdf) const {
        const SGEnvLeaf &leaf = leaves[leafIndex];
        Float phi = Lerp(std::min(u0, OneMinusEpsilon), leaf.phi0, leaf.phi1);
        Float cosTheta =
            Lerp(std::min(u1, OneMinusEpsilon), leaf.cosTheta1, leaf.cosTheta0);
        Float theta = std::acos(Clamp(cosTheta, Float(-1), Float(1)));
        *localDir = LatLongDirection(phi, theta);
        *pdf = leaf.solidAngle > 0 ? 1 / leaf.solidAngle : 0;
        return Point2f(phi * Inv2Pi, theta * InvPi);
    }

    Float Pdf(const Interaction *ref, const Transform &LightToWorld,
              const Vector3f &localDir) const {
        if (Empty()) return 0;
        int leafIndex = LeafForDirection(localDir);
        if (leafIndex < 0) return 0;
        Float leafPdf = LeafProbability(root, leafIndex, ref, LightToWorld);
        const SGEnvLeaf &leaf = leaves[leafIndex];
        return leaf.solidAngle > 0 ? leafPdf / leaf.solidAngle : 0;
    }

  private:
    int Build(int tx0, int tx1, int ty0, int ty1) {
        int nodeIndex = nodes.size();
        nodes.push_back(SGEnvNode());
        nodes[nodeIndex].tx0 = tx0;
        nodes[nodeIndex].tx1 = tx1;
        nodes[nodeIndex].ty0 = ty0;
        nodes[nodeIndex].ty1 = ty1;
        if (tx1 - tx0 == 1 && ty1 - ty0 == 1) {
            nodes[nodeIndex].leaf = AddLeaf(tx0, ty0);
        } else if (tx1 - tx0 >= ty1 - ty0) {
            int mid = (tx0 + tx1) / 2;
            int left = Build(tx0, mid, ty0, ty1);
            int right = Build(mid, tx1, ty0, ty1);
            nodes[nodeIndex].left = left;
            nodes[nodeIndex].right = right;
        } else {
            int mid = (ty0 + ty1) / 2;
            int left = Build(tx0, tx1, ty0, mid);
            int right = Build(tx0, tx1, mid, ty1);
            nodes[nodeIndex].left = left;
            nodes[nodeIndex].right = right;
        }

        SGEnvNode &node = nodes[nodeIndex];
        if (node.IsLeaf()) {
            const SGEnvLeaf &leaf = leaves[node.leaf];
            node.flux = leaf.flux;
            node.solidAngle = leaf.solidAngle;
            node.axis = leaf.axis;
            node.moment = leaf.axis * (leaf.flux > 0 ? leaf.flux
                                                     : leaf.solidAngle);
            node.momentWeight = leaf.flux > 0 ? leaf.flux : leaf.solidAngle;
        } else {
            const SGEnvNode &left = nodes[node.left];
            const SGEnvNode &right = nodes[node.right];
            node.flux = left.flux + right.flux;
            node.solidAngle = left.solidAngle + right.solidAngle;
            node.moment = left.moment + right.moment;
            node.momentWeight = left.momentWeight + right.momentWeight;
        }
        FinalizeSGNode(&node);
        return nodeIndex;
    }

    int AddLeaf(int tx, int ty) {
        SGEnvLeaf leaf;
        leaf.tx = tx;
        leaf.ty = ty;
        leaf.u0 = tx * tileSize;
        leaf.u1 = std::min(width, leaf.u0 + tileSize);
        leaf.v0 = ty * tileSize;
        leaf.v1 = std::min(height, leaf.v0 + tileSize);
        leaf.phi0 = 2 * Pi * leaf.u0 / width;
        leaf.phi1 = 2 * Pi * leaf.u1 / width;
        Float theta0 = Pi * leaf.v0 / height;
        Float theta1 = Pi * leaf.v1 / height;
        leaf.cosTheta0 = std::cos(theta0);
        leaf.cosTheta1 = std::cos(theta1);
        leaf.solidAngle =
            SphericalRectSolidAngle(leaf.phi0, leaf.phi1, theta0, theta1);

        Vector3f weightedMoment(0, 0, 0);
        Vector3f areaMoment(0, 0, 0);
        Float fwidth = 0.5f / std::min(width, height);
        for (int v = leaf.v0; v < leaf.v1; ++v) {
            Float vp = (v + Float(0.5)) / height;
            Float theta = vp * Pi;
            Float rowTheta0 = Pi * v / height;
            Float rowTheta1 = Pi * (v + 1) / height;
            for (int u = leaf.u0; u < leaf.u1; ++u) {
                Float up = (u + Float(0.5)) / width;
                Float phi = up * 2 * Pi;
                Float cellSolidAngle = SphericalRectSolidAngle(
                    2 * Pi * u / width, 2 * Pi * (u + 1) / width, rowTheta0,
                    rowTheta1);
                Vector3f dir = LatLongDirection(phi, theta);
                Float luminance =
                    SpectrumY(Spectrum(lmap.Lookup(Point2f(up, vp), fwidth),
                                       SpectrumType::Illuminant));
                Float flux = luminance * cellSolidAngle;
                leaf.flux += flux;
                weightedMoment += flux * dir;
                areaMoment += cellSolidAngle * dir;
            }
        }

        Vector3f moment = leaf.flux > 0 ? weightedMoment : areaMoment;
        Float len = moment.Length();
        if (len > 0) leaf.axis = moment / len;
        int leafIndex = leaves.size();
        leaves.push_back(leaf);
        leafForTile[tx + ty * tilesX] = leafIndex;
        return leafIndex;
    }

    Float Importance(const SGEnvNode &node, const Interaction *ref,
                     const Transform &LightToWorld) const {
        if (node.flux <= 0) return 0;
        Float amplitude =
            node.flux / std::max(SGIntegral(node.sharpness), Float(MachineEpsilon));
        Vector3f axis = LightToWorld(node.axis);
        Float importance = 0;
        if (ref && ref->IsSurfaceInteraction()) {
            const SurfaceInteraction &si = (const SurfaceInteraction &)*ref;
            if (si.bsdf) {
                Spectrum diffuseWeight, glossyWeight;
                Float alphax, alphay;
                si.bsdf->SGSamplingInfo(si.wo, &diffuseWeight, &glossyWeight,
                                        &alphax, &alphay);
                importance += SGDiffuseImportance(axis, node.sharpness, amplitude,
                                                  si, diffuseWeight);
                importance += SGGlossyImportance(axis, node.sharpness, amplitude,
                                                 si, glossyWeight, alphax,
                                                 alphay);
            }
        } else {
            importance = node.flux * Inv4Pi;
        }

        Float floor = Float(1e-6) * node.flux * Inv4Pi;
        return std::max(importance, floor);
    }

    Float ChildProbability(const SGEnvNode &node, const Interaction *ref,
                           const Transform &LightToWorld) const {
        Float leftImportance = Importance(nodes[node.left], ref, LightToWorld);
        Float rightImportance = Importance(nodes[node.right], ref, LightToWorld);
        Float sum = leftImportance + rightImportance;
        return sum > 0 ? Clamp(leftImportance / sum, Float(0), Float(1))
                       : Float(0.5);
    }

    Float LeafProbability(int nodeIndex, int leafIndex, const Interaction *ref,
                          const Transform &LightToWorld) const {
        const SGEnvNode &node = nodes[nodeIndex];
        if (node.IsLeaf()) return node.leaf == leafIndex ? 1 : 0;

        const SGEnvLeaf &leaf = leaves[leafIndex];
        Float pLeft = ChildProbability(node, ref, LightToWorld);
        const SGEnvNode &left = nodes[node.left];
        bool inLeft = leaf.tx >= left.tx0 && leaf.tx < left.tx1 &&
                      leaf.ty >= left.ty0 && leaf.ty < left.ty1;
        if (inLeft)
            return pLeft * LeafProbability(node.left, leafIndex, ref, LightToWorld);
        return (1 - pLeft) *
               LeafProbability(node.right, leafIndex, ref, LightToWorld);
    }

    int LeafForDirection(const Vector3f &localDir) const {
        Float theta = SphericalTheta(localDir), phi = SphericalPhi(localDir);
        int u = Clamp(int(phi * Inv2Pi * width), 0, width - 1);
        int v = Clamp(int(theta * InvPi * height), 0, height - 1);
        int tx = std::min(u / tileSize, tilesX - 1);
        int ty = std::min(v / tileSize, tilesY - 1);
        return leafForTile[tx + ty * tilesX];
    }

    const MIPMap<RGBSpectrum> &lmap;
    int width, height, tileSize;
    int tilesX, tilesY;
    std::vector<int> leafForTile;
    std::vector<SGEnvNode> nodes;
    std::vector<SGEnvLeaf> leaves;
    int root = -1;
};

SGInfiniteAreaLight::SGInfiniteAreaLight(const Transform &LightToWorld,
                                         const Spectrum &L, int nSamples,
                                         const std::string &texmap,
                                         int tileSize,
                                         Float treeSamplingWeight)
    : Light((int)LightFlags::Infinite, LightToWorld, MediumInterface(),
            nSamples),
      treeSamplingWeight(Clamp(treeSamplingWeight, Float(0), Float(1))) {
    Point2i resolution;
    std::unique_ptr<RGBSpectrum[]> texels(nullptr);
    if (texmap != "") {
        texels = ReadImage(texmap, &resolution);
        if (texels)
            for (int i = 0; i < resolution.x * resolution.y; ++i)
                texels[i] *= L.ToRGBSpectrum();
    }
    if (!texels) {
        resolution.x = resolution.y = 1;
        texels = std::unique_ptr<RGBSpectrum[]>(new RGBSpectrum[1]);
        texels[0] = L.ToRGBSpectrum();
    }
    Lmap.reset(new MIPMap<RGBSpectrum>(resolution, texels.get()));

    int width = 2 * Lmap->Width(), height = 2 * Lmap->Height();
    std::unique_ptr<Float[]> img(new Float[width * height]);
    float fwidth = 0.5f / std::min(width, height);
    ParallelFor(
        [&](int64_t v) {
            Float vp = (v + .5f) / (Float)height;
            Float sinTheta = std::sin(Pi * (v + .5f) / height);
            for (int u = 0; u < width; ++u) {
                Float up = (u + .5f) / (Float)width;
                img[u + v * width] = Lmap->Lookup(Point2f(up, vp), fwidth).y();
                img[u + v * width] *= sinTheta;
            }
        },
        height, 32);
    distribution.reset(new Distribution2D(img.get(), width, height));
    tree.reset(new SGEnvLightTree(*Lmap, width, height, tileSize));
    if (tree->Empty()) this->treeSamplingWeight = 0;
}

SGInfiniteAreaLight::~SGInfiniteAreaLight() {}

Spectrum SGInfiniteAreaLight::Power() const {
    return Pi * worldRadius * worldRadius *
           Spectrum(Lmap->Lookup(Point2f(.5f, .5f), .5f),
                    SpectrumType::Illuminant);
}

Spectrum SGInfiniteAreaLight::Le(const RayDifferential &ray) const {
    Vector3f w = Normalize(WorldToLight(ray.d));
    Point2f st(SphericalPhi(w) * Inv2Pi, SphericalTheta(w) * InvPi);
    return Spectrum(Lmap->Lookup(st), SpectrumType::Illuminant);
}

Spectrum SGInfiniteAreaLight::Sample_Li(const Interaction &ref, const Point2f &u,
                                        Vector3f *wi, Float *pdf,
                                        VisibilityTester *vis) const {
    ProfilePhase _(Prof::LightSample);
    Point2f uv;
    Vector3f localDir;
    if (tree && treeSamplingWeight > 0 && u[0] < treeSamplingWeight) {
        int leafIndex;
        Float leafPdf, uRemapped;
        if (!tree->Sample(&ref, LightToWorld, u[0] / treeSamplingWeight,
                          &leafIndex, &leafPdf, &uRemapped) ||
            leafPdf == 0) {
            *pdf = 0;
            return Spectrum(0.f);
        }
        Float unusedPdf;
        uv = tree->SampleLeaf(leafIndex, uRemapped, u[1], &localDir, &unusedPdf);
    } else {
        Float mapPdf;
        Float u0 = treeSamplingWeight < 1
                       ? (u[0] - treeSamplingWeight) / (1 - treeSamplingWeight)
                       : u[0];
        uv = distribution->SampleContinuous(Point2f(u0, u[1]), &mapPdf);
        if (mapPdf == 0) {
            *pdf = 0;
            return Spectrum(0.f);
        }
        Float theta = uv[1] * Pi, phi = uv[0] * 2 * Pi;
        localDir = LatLongDirection(phi, theta);
    }

    *wi = LightToWorld(localDir);
    *pdf = DirectionPdf(&ref, *wi);
    if (*pdf == 0) return Spectrum(0.f);
    *vis = VisibilityTester(ref, Interaction(ref.p + *wi * (2 * worldRadius),
                                             ref.time, mediumInterface));
    return Spectrum(Lmap->Lookup(uv), SpectrumType::Illuminant);
}

Float SGInfiniteAreaLight::DirectionPdf(const Interaction *ref,
                                        const Vector3f &w) const {
    Vector3f wi = Normalize(WorldToLight(w));
    Float theta = SphericalTheta(wi), phi = SphericalPhi(wi);
    Point2f uv(phi * Inv2Pi, theta * InvPi);

    Float basePdf = 0;
    Float sinTheta = std::sin(theta);
    if (sinTheta > 0)
        basePdf = distribution->Pdf(uv) / (2 * Pi * Pi * sinTheta);

    Float treePdf = tree ? tree->Pdf(ref, LightToWorld, wi) : 0;
    return treeSamplingWeight * treePdf +
           (1 - treeSamplingWeight) * basePdf;
}

Float SGInfiniteAreaLight::Pdf_Li(const Interaction &ref,
                                  const Vector3f &w) const {
    ProfilePhase _(Prof::LightPdf);
    return DirectionPdf(&ref, w);
}

Spectrum SGInfiniteAreaLight::Sample_Le(const Point2f &u1, const Point2f &u2,
                                        Float time, Ray *ray,
                                        Normal3f *nLight, Float *pdfPos,
                                        Float *pdfDir) const {
    ProfilePhase _(Prof::LightSample);
    Point2f uv;
    Vector3f localDir;
    if (tree && treeSamplingWeight > 0 && u1[0] < treeSamplingWeight) {
        int leafIndex;
        Float leafPdf, uRemapped;
        if (!tree->Sample(nullptr, LightToWorld, u1[0] / treeSamplingWeight,
                          &leafIndex, &leafPdf, &uRemapped) ||
            leafPdf == 0)
            return Spectrum(0.f);
        Float unusedPdf;
        uv = tree->SampleLeaf(leafIndex, uRemapped, u1[1], &localDir,
                              &unusedPdf);
    } else {
        Float mapPdf;
        Float u0 = treeSamplingWeight < 1
                       ? (u1[0] - treeSamplingWeight) / (1 - treeSamplingWeight)
                       : u1[0];
        uv = distribution->SampleContinuous(Point2f(u0, u1[1]), &mapPdf);
        if (mapPdf == 0) return Spectrum(0.f);
        Float theta = uv[1] * Pi, phi = uv[0] * 2.f * Pi;
        localDir = LatLongDirection(phi, theta);
    }

    Vector3f d = -LightToWorld(localDir);
    *nLight = (Normal3f)d;

    Vector3f v1, v2;
    CoordinateSystem(-d, &v1, &v2);
    Point2f cd = ConcentricSampleDisk(u2);
    Point3f pDisk = worldCenter + worldRadius * (cd.x * v1 + cd.y * v2);
    *ray = Ray(pDisk + worldRadius * -d, d, Infinity, time);

    *pdfDir = DirectionPdf(nullptr, -d);
    *pdfPos = 1 / (Pi * worldRadius * worldRadius);
    return Spectrum(Lmap->Lookup(uv), SpectrumType::Illuminant);
}

void SGInfiniteAreaLight::Pdf_Le(const Ray &ray, const Normal3f &, Float *pdfPos,
                                 Float *pdfDir) const {
    ProfilePhase _(Prof::LightPdf);
    *pdfDir = DirectionPdf(nullptr, -ray.d);
    *pdfPos = 1 / (Pi * worldRadius * worldRadius);
}

std::shared_ptr<SGInfiniteAreaLight> CreateSGInfiniteLight(
    const Transform &light2world, const ParamSet &paramSet) {
    Spectrum L = paramSet.FindOneSpectrum("L", Spectrum(1.0));
    Spectrum sc = paramSet.FindOneSpectrum("scale", Spectrum(1.0));
    std::string texmap = paramSet.FindOneFilename("mapname", "");
    int nSamples = paramSet.FindOneInt("samples",
                                       paramSet.FindOneInt("nsamples", 1));
    int tileSize = paramSet.FindOneInt("sgtilesize", 4);
    Float treeWeight = paramSet.FindOneFloat("sgtreeweight", Float(0.9));
    if (PbrtOptions.quickRender) nSamples = std::max(1, nSamples / 4);
    return std::make_shared<SGInfiniteAreaLight>(light2world, L * sc, nSamples,
                                                 texmap, tileSize, treeWeight);
}

}  // namespace pbrt
