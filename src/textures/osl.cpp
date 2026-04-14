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

// textures/osl.cpp*
#include "textures/osl.h"

#include "interaction.h"
#include "shape.h"

#include <atomic>
#include <cstdarg>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef PBRT_ENABLE_OSL
#include <OSL/genclosure.h>
#include <OSL/rendererservices.h>
#endif

namespace pbrt {

namespace {

std::atomic<bool> gMissingShaderWarned(false);
std::atomic<bool> gMissingOutputWarned(false);
std::atomic<bool> gOSLDisabledWarned(false);
std::atomic<bool> gGroupspecWithShaderWarned(false);
std::atomic<bool> gInvalidConfigWarned(false);
std::atomic<bool> gUnsupportedRendererCallWarned(false);

bool OSLDebugEnabled() {
    static bool enabled = (std::getenv("PBRT_OSL_DEBUG") != nullptr);
    return enabled;
}

void OSLLog(const char *fmt, ...) {
    if (!OSLDebugEnabled()) return;
    va_list args;
    va_start(args, fmt);
    std::string msg = StringVaprintf(fmt, args);
    va_end(args);
    Warning("[OSL][debug] %s", msg.c_str());
}

bool WarnUnsupportedRendererCall(const char *what) {
    if (!gUnsupportedRendererCallWarned.exchange(true))
        Warning("[OSL][renderer] Unsupported renderer callback: %s", what);
    return false;
}

bool IsEmpty(const std::string &s) { return s.empty(); }

#ifndef PBRT_ENABLE_OSL
void WarnIfOSLDisabled() {
    if (!gOSLDisabledWarned.exchange(true))
        Warning("[OSL] OSL support is disabled. Configure with PBRT_ENABLE_OSL=ON.");
}
#endif

}  // namespace

bool ValidateAndNormalizeOSLShaderConfig(OSLShaderConfig *config,
                                         const char *ownerLabel) {
    if (!config) return false;
    if (IsEmpty(config->outputName)) config->outputName = "result";
    if (!IsEmpty(config->shader) && !IsEmpty(config->groupSpec) &&
        !gGroupspecWithShaderWarned.exchange(true)) {
        Warning(
            "[OSL][contract] %s provides both \"shader\" and \"groupspec\"; "
            "using groupspec and ignoring shader.",
            ownerLabel ? ownerLabel : "osl");
        config->shader.clear();
    }
    if (!IsEmpty(config->layer) && IsEmpty(config->shader) &&
        IsEmpty(config->groupSpec)) {
        if (!gInvalidConfigWarned.exchange(true))
            Warning(
                "[OSL][contract] %s sets \"layer\" but missing \"shader\" and "
                "\"groupspec\".",
                ownerLabel ? ownerLabel : "osl");
        return false;
    }
    if (IsEmpty(config->shader) && IsEmpty(config->groupSpec)) {
        if (!gMissingShaderWarned.exchange(true))
            Warning("[OSL][contract] %s requires \"shader\" or \"groupspec\".",
                    ownerLabel ? ownerLabel : "osl");
        return false;
    }
    return true;
}

#ifdef PBRT_ENABLE_OSL
namespace {

enum ClosureIDs {
    EMISSION_ID = 1,
    DIFFUSE_ID,
    OREN_NAYAR_ID,
    MICROFACET_ID,
    REFLECTION_ID,
    FRESNEL_REFLECTION_ID,
    REFRACTION_ID,
    TRANSPARENT_ID,
    DIELECTRIC_BSDF_ID,
    CONDUCTOR_BSDF_ID,
    TRANSPARENT_BSDF_ID
};

struct EmptyParams {};
struct DiffuseParams {
    OSL::Vec3 N;
};
struct OrenNayarParams {
    OSL::Vec3 N;
    float sigma;
};
struct MicrofacetParams {
    OSL::ustringhash dist;
    OSL::Vec3 N;
    OSL::Vec3 U;
    float xalpha, yalpha, eta;
    int refract;
};
struct ReflectionParams {
    OSL::Vec3 N;
    float eta;
};
struct RefractionParams {
    OSL::Vec3 N;
    float eta;
};
struct DielectricBsdfParams {
    OSL::Vec3 N, U;
    OSL::Color3 reflection_tint;
    OSL::Color3 transmission_tint;
    float roughness_x, roughness_y;
    float ior;
    OSL::ustringhash distribution;
};
struct ConductorBsdfParams {
    OSL::Vec3 N, U;
    float roughness_x, roughness_y;
    OSL::Color3 ior;
    OSL::Color3 extinction;
    OSL::ustringhash distribution;
};

struct OSLRenderState {
    const SurfaceInteraction *si = nullptr;
};

struct OSLRuntimeCountersAtomic {
    std::atomic<uint64_t> executions{0};
    std::atomic<uint64_t> groupCacheHits{0};
    std::atomic<uint64_t> groupCacheMisses{0};
    std::atomic<uint64_t> symbolCacheHits{0};
    std::atomic<uint64_t> symbolCacheMisses{0};
    std::atomic<uint64_t> executeFailures{0};
    std::atomic<uint64_t> fallbackCount{0};
};

OSLRuntimeCountersAtomic gCounters;

struct ThreadResources {
    OSL::PerThreadInfo *threadInfo = nullptr;
    OSL::ShadingContext *context = nullptr;
};

class PbrtOSLRendererServices : public OSL::RendererServices {
  public:
    PbrtOSLRendererServices() : OSL::RendererServices(nullptr) {}

    bool get_matrix(OSL::ShaderGlobals *sg, OSL::Matrix44 &result,
                    OSL::TransformationPtr xform, float time) override {
        return SetMatrixFromTransformPtr(result, xform, false);
    }

    bool get_matrix(OSL::ShaderGlobals *sg, OSL::Matrix44 &result,
                    OSL::TransformationPtr xform) override {
        return SetMatrixFromTransformPtr(result, xform, false);
    }

    bool get_inverse_matrix(OSL::ShaderGlobals *sg, OSL::Matrix44 &result,
                            OSL::TransformationPtr xform,
                            float time) override {
        return SetMatrixFromTransformPtr(result, xform, true);
    }

    bool get_inverse_matrix(OSL::ShaderGlobals *sg, OSL::Matrix44 &result,
                            OSL::TransformationPtr xform) override {
        return SetMatrixFromTransformPtr(result, xform, true);
    }

    bool get_matrix(OSL::ShaderGlobals *sg, OSL::Matrix44 &result,
                    OSL::ustringhash from, float time) override {
        if (from == OSL::ustringhash("common") || from == OSL::ustringhash(""))
            return SetIdentity(result);
        if (from == OSL::ustringhash("object"))
            return SetMatrixFromTransformPtr(result, sg->object2common, false);
        if (from == OSL::ustringhash("shader"))
            return SetMatrixFromTransformPtr(result, sg->shader2common, false);
        return WarnUnsupportedRendererCall("get_matrix(from)");
    }

    bool get_matrix(OSL::ShaderGlobals *sg, OSL::Matrix44 &result,
                    OSL::ustringhash from) override {
        return get_matrix(sg, result, from, sg ? sg->time : 0.f);
    }

    bool get_inverse_matrix(OSL::ShaderGlobals *sg, OSL::Matrix44 &result,
                            OSL::ustringhash to, float time) override {
        if (to == OSL::ustringhash("common") || to == OSL::ustringhash(""))
            return SetIdentity(result);
        if (to == OSL::ustringhash("object"))
            return SetMatrixFromTransformPtr(result, sg->object2common, true);
        if (to == OSL::ustringhash("shader"))
            return SetMatrixFromTransformPtr(result, sg->shader2common, true);
        return WarnUnsupportedRendererCall("get_inverse_matrix(to)");
    }

    bool get_inverse_matrix(OSL::ShaderGlobals *sg, OSL::Matrix44 &result,
                            OSL::ustringhash to) override {
        return get_inverse_matrix(sg, result, to, sg ? sg->time : 0.f);
    }

    bool get_userdata(bool derivatives, OSL::ustringhash name, OSL::TypeDesc type,
                      OSL::ShaderGlobals *sg, void *val) override {
        if (!sg || !val) return false;
        const OSLRenderState *rs = reinterpret_cast<const OSLRenderState *>(sg->renderstate);
        if (!rs || !rs->si) return false;
        const SurfaceInteraction &si = *rs->si;
        if (type == OSL::TypeFloat && name == OSL::ustringhash("u")) {
            *reinterpret_cast<float *>(val) = si.uv.x;
            return true;
        }
        if (type == OSL::TypeFloat && name == OSL::ustringhash("v")) {
            *reinterpret_cast<float *>(val) = si.uv.y;
            return true;
        }
        if (type == OSL::TypeVector && name == OSL::ustringhash("N")) {
            float *v = reinterpret_cast<float *>(val);
            v[0] = si.shading.n.x;
            v[1] = si.shading.n.y;
            v[2] = si.shading.n.z;
            return true;
        }
        if (type == OSL::TypePoint && name == OSL::ustringhash("P")) {
            float *v = reinterpret_cast<float *>(val);
            v[0] = si.p.x;
            v[1] = si.p.y;
            v[2] = si.p.z;
            return true;
        }
        return false;
    }

    bool get_attribute(OSL::ShaderGlobals *sg, bool derivatives,
                       OSL::ustringhash object, OSL::TypeDesc type,
                       OSL::ustringhash name, void *val) override {
        return get_userdata(derivatives, name, type, sg, val);
    }

  private:
    static bool SetIdentity(OSL::Matrix44 &result) {
        result.makeIdentity();
        return true;
    }

    static bool SetMatrixFromTransformPtr(OSL::Matrix44 &result,
                                          OSL::TransformationPtr xform,
                                          bool inverse) {
        if (!xform) return SetIdentity(result);
        const Transform *t = reinterpret_cast<const Transform *>(xform);
        const Matrix4x4 &m = inverse ? t->GetInverseMatrix() : t->GetMatrix();
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) result[i][j] = m.m[i][j];
        return true;
    }
};

class OSLRuntime {
  public:
    OSLRuntime()
        : renderer(new PbrtOSLRendererServices()),
          ss(new OSL::ShadingSystem(renderer.get(), nullptr, nullptr)) {
        RegisterClosures();
    }

    ~OSLRuntime() {
        std::lock_guard<std::mutex> lock(threadMutex);
        for (auto &kv : threadData) {
            if (kv.second.context) ss->release_context(kv.second.context);
            if (kv.second.threadInfo) ss->destroy_thread_info(kv.second.threadInfo);
        }
        threadData.clear();
    }

    OSL::ShadingContext *GetContext() {
        const auto tid = std::this_thread::get_id();
        std::lock_guard<std::mutex> lock(threadMutex);
        ThreadResources &res = threadData[tid];
        if (!res.threadInfo) res.threadInfo = ss->create_thread_info();
        if (!res.context) res.context = ss->get_context(res.threadInfo);
        return res.context;
    }

    OSL::ShaderGroupRef ResolveGroup(const OSLShaderConfig &config) {
        const std::string key = config.shader + "|" + config.group + "|" +
                                config.layer + "|" + config.groupSpec;
        std::lock_guard<std::mutex> lock(cacheMutex);
        auto iter = groups.find(key);
        if (iter != groups.end()) {
            ++gCounters.groupCacheHits;
            return iter->second;
        }
        ++gCounters.groupCacheMisses;
        OSL::ShaderGroupRef groupRef = BuildGroup(config);
        if (groupRef) groups.emplace(key, groupRef);
        return groupRef;
    }

    const OSL::ShaderSymbol *ResolveOutputSymbol(const OSLShaderConfig &config,
                                                 const OSL::ShaderGroup &group) {
        const std::string key = StringPrintf("%p|%s|%s", &group,
                                             config.layer.c_str(),
                                             config.outputName.c_str());
        std::lock_guard<std::mutex> lock(cacheMutex);
        auto it = outputSymbols.find(key);
        if (it != outputSymbols.end()) {
            ++gCounters.symbolCacheHits;
            return it->second;
        }
        ++gCounters.symbolCacheMisses;
        const OSL::ShaderSymbol *sym = nullptr;
        if (!config.layer.empty())
            sym = ss->find_symbol(group, OSL::ustring(config.layer),
                                  OSL::ustring(config.outputName));
        if (!sym) sym = ss->find_symbol(group, OSL::ustring(config.outputName));
        outputSymbols[key] = sym;
        return sym;
    }

    int GetClosureId(const char *name) {
        if (!name) return -1;
        int id = -1;
        const char *queryName = name;
        const OSL::ClosureParam *params = nullptr;
        if (ss->query_closure(&queryName, &id, &params)) return id;
        return -1;
    }

    OSL::ShadingSystem *System() { return ss.get(); }
    OSL::RendererServices *Renderer() { return renderer.get(); }

  private:
    OSL::ShaderGroupRef BuildGroup(const OSLShaderConfig &config) {
        const std::string groupName =
            config.group.empty()
                ? (config.shader.empty() ? "pbrt_osl_group" : config.shader)
                : config.group;
        OSL::ShaderGroupRef groupRef;
        if (!config.groupSpec.empty()) {
            groupRef = ss->ShaderGroupBegin(groupName, "surface", config.groupSpec);
            if (!groupRef) return OSL::ShaderGroupRef();
            if (!ss->ShaderGroupEnd(*groupRef)) return OSL::ShaderGroupRef();
        } else {
            if (config.shader.empty()) return OSL::ShaderGroupRef();
            groupRef = ss->ShaderGroupBegin(groupName);
            if (!groupRef) return OSL::ShaderGroupRef();
            const std::string layerName =
                config.layer.empty() ? "pbrt_osl_layer" : config.layer;
            if (!ss->Shader(*groupRef, "surface", config.shader, layerName))
                return OSL::ShaderGroupRef();
            if (!ss->ShaderGroupEnd(*groupRef)) return OSL::ShaderGroupRef();
        }
        std::vector<const char *> outputs;
        outputs.push_back("Ci");
        if (!config.outputName.empty()) outputs.push_back(config.outputName.c_str());
        ss->attribute(groupRef.get(), "renderer_outputs",
                      OSL::TypeDesc(OSL::TypeDesc::STRING, int(outputs.size())),
                      outputs.data());
        ss->optimize_group(groupRef.get(), GetContext(), true);
        OSLLog("group built name=%s shader=%s layer=%s", groupName.c_str(),
               config.shader.c_str(), config.layer.c_str());
        return groupRef;
    }

    std::unique_ptr<PbrtOSLRendererServices> renderer;
    std::unique_ptr<OSL::ShadingSystem> ss;
    std::mutex threadMutex;
    std::unordered_map<std::thread::id, ThreadResources> threadData;
    std::mutex cacheMutex;
    std::unordered_map<std::string, OSL::ShaderGroupRef> groups;
    std::unordered_map<std::string, const OSL::ShaderSymbol *> outputSymbols;

    void RegisterClosures() {
        static OSL::ClosureParam emissionParams[] = {
            CLOSURE_FINISH_PARAM(EmptyParams),
        };
        static OSL::ClosureParam diffuseParams[] = {
            CLOSURE_VECTOR_PARAM(DiffuseParams, N),
            CLOSURE_FINISH_PARAM(DiffuseParams),
        };
        static OSL::ClosureParam orenParams[] = {
            CLOSURE_VECTOR_PARAM(OrenNayarParams, N),
            CLOSURE_FLOAT_PARAM(OrenNayarParams, sigma),
            CLOSURE_FINISH_PARAM(OrenNayarParams),
        };
        static OSL::ClosureParam microfacetParams[] = {
            CLOSURE_STRING_PARAM(MicrofacetParams, dist),
            CLOSURE_VECTOR_PARAM(MicrofacetParams, N),
            CLOSURE_VECTOR_PARAM(MicrofacetParams, U),
            CLOSURE_FLOAT_PARAM(MicrofacetParams, xalpha),
            CLOSURE_FLOAT_PARAM(MicrofacetParams, yalpha),
            CLOSURE_FLOAT_PARAM(MicrofacetParams, eta),
            CLOSURE_INT_PARAM(MicrofacetParams, refract),
            CLOSURE_FINISH_PARAM(MicrofacetParams),
        };
        static OSL::ClosureParam reflectionParams[] = {
            CLOSURE_VECTOR_PARAM(ReflectionParams, N),
            CLOSURE_FINISH_PARAM(ReflectionParams),
        };
        static OSL::ClosureParam fresnelReflectionParams[] = {
            CLOSURE_VECTOR_PARAM(ReflectionParams, N),
            CLOSURE_FLOAT_PARAM(ReflectionParams, eta),
            CLOSURE_FINISH_PARAM(ReflectionParams),
        };
        static OSL::ClosureParam refractionParams[] = {
            CLOSURE_VECTOR_PARAM(RefractionParams, N),
            CLOSURE_FLOAT_PARAM(RefractionParams, eta),
            CLOSURE_FINISH_PARAM(RefractionParams),
        };
        static OSL::ClosureParam transparentParams[] = {
            CLOSURE_FINISH_PARAM(EmptyParams),
        };
        static OSL::ClosureParam dielectricBsdfParams[] = {
            CLOSURE_VECTOR_PARAM(DielectricBsdfParams, N),
            CLOSURE_VECTOR_PARAM(DielectricBsdfParams, U),
            CLOSURE_COLOR_PARAM(DielectricBsdfParams, reflection_tint),
            CLOSURE_COLOR_PARAM(DielectricBsdfParams, transmission_tint),
            CLOSURE_FLOAT_PARAM(DielectricBsdfParams, roughness_x),
            CLOSURE_FLOAT_PARAM(DielectricBsdfParams, roughness_y),
            CLOSURE_FLOAT_PARAM(DielectricBsdfParams, ior),
            CLOSURE_STRING_PARAM(DielectricBsdfParams, distribution),
            CLOSURE_FINISH_PARAM(DielectricBsdfParams),
        };
        static OSL::ClosureParam conductorBsdfParams[] = {
            CLOSURE_VECTOR_PARAM(ConductorBsdfParams, N),
            CLOSURE_VECTOR_PARAM(ConductorBsdfParams, U),
            CLOSURE_FLOAT_PARAM(ConductorBsdfParams, roughness_x),
            CLOSURE_FLOAT_PARAM(ConductorBsdfParams, roughness_y),
            CLOSURE_COLOR_PARAM(ConductorBsdfParams, ior),
            CLOSURE_COLOR_PARAM(ConductorBsdfParams, extinction),
            CLOSURE_STRING_PARAM(ConductorBsdfParams, distribution),
            CLOSURE_FINISH_PARAM(ConductorBsdfParams),
        };

        ss->register_closure("emission", EMISSION_ID, emissionParams, nullptr,
                             nullptr);
        ss->register_closure("diffuse", DIFFUSE_ID, diffuseParams, nullptr,
                             nullptr);
        ss->register_closure("oren_nayar", OREN_NAYAR_ID, orenParams, nullptr,
                             nullptr);
        ss->register_closure("microfacet", MICROFACET_ID, microfacetParams,
                             nullptr, nullptr);
        ss->register_closure("reflection", REFLECTION_ID, reflectionParams,
                             nullptr, nullptr);
        ss->register_closure("reflection", FRESNEL_REFLECTION_ID,
                             fresnelReflectionParams, nullptr, nullptr);
        ss->register_closure("refraction", REFRACTION_ID, refractionParams,
                             nullptr, nullptr);
        ss->register_closure("transparent", TRANSPARENT_ID, transparentParams,
                             nullptr, nullptr);
        ss->register_closure("dielectric_bsdf", DIELECTRIC_BSDF_ID,
                             dielectricBsdfParams, nullptr, nullptr);
        ss->register_closure("conductor_bsdf", CONDUCTOR_BSDF_ID,
                             conductorBsdfParams, nullptr, nullptr);
        ss->register_closure("transparent_bsdf", TRANSPARENT_BSDF_ID,
                             transparentParams, nullptr, nullptr);
    }
};

OSLRuntime &GetRuntime() {
    static OSLRuntime runtime;
    return runtime;
}

OSL::Vec3 ToOSL(const Vector3f &v) { return OSL::Vec3(v.x, v.y, v.z); }
OSL::Vec3 ToOSL(const Point3f &p) { return OSL::Vec3(p.x, p.y, p.z); }

void InitializeShaderGlobals(const SurfaceInteraction &si,
                             OSL::RendererServices *renderer,
                             OSLRenderState *renderState,
                             OSL::ShaderGlobals *sg) {
    memset(sg, 0, sizeof(*sg));
    sg->P = ToOSL(si.p);
    sg->dPdx = ToOSL(si.dpdx);
    sg->dPdy = ToOSL(si.dpdy);
    sg->I = ToOSL(-si.wo);
    sg->N = ToOSL(Vector3f(si.shading.n));
    sg->Ng = ToOSL(Vector3f(si.n));
    sg->u = si.uv.x;
    sg->v = si.uv.y;
    sg->dudx = si.dudx;
    sg->dudy = si.dudy;
    sg->dvdx = si.dvdx;
    sg->dvdy = si.dvdy;
    sg->dPdu = ToOSL(si.shading.dpdu);
    sg->dPdv = ToOSL(si.shading.dpdv);
    sg->time = si.time;
    sg->renderer = renderer;
    sg->Ci = nullptr;
    sg->backfacing = Dot(si.wo, Vector3f(si.n)) < 0.f ? 1 : 0;
    renderState->si = &si;
    sg->renderstate = renderState;
    if (si.shape) {
        sg->object2common = si.shape->ObjectToWorld;
        sg->shader2common = si.shape->ObjectToWorld;
    }
}

bool ExtractFloat(const OSL::TypeDesc &type, const void *address, Float *out) {
    if (!address) return false;
    if (type.basetype == OSL::TypeDesc::FLOAT &&
        type.aggregate == OSL::TypeDesc::SCALAR) {
        *out = *(const float *)address;
        return true;
    }
    if (type.basetype == OSL::TypeDesc::INT &&
        type.aggregate == OSL::TypeDesc::SCALAR) {
        *out = Float(*(const int *)address);
        return true;
    }
    if (type.basetype == OSL::TypeDesc::FLOAT &&
        type.aggregate == OSL::TypeDesc::VEC3) {
        *out = ((const float *)address)[0];
        return true;
    }
    return false;
}

bool ExtractSpectrum(const OSL::TypeDesc &type, const void *address,
                     Spectrum *out) {
    if (!address) return false;
    if (type.basetype == OSL::TypeDesc::FLOAT &&
        type.aggregate == OSL::TypeDesc::VEC3) {
        const float *v = (const float *)address;
        Float rgb[3] = {Float(v[0]), Float(v[1]), Float(v[2])};
        *out = Spectrum::FromRGB(rgb);
        return true;
    }
    if (type.basetype == OSL::TypeDesc::FLOAT &&
        type.aggregate == OSL::TypeDesc::SCALAR) {
        *out = Spectrum(*(const float *)address);
        return true;
    }
    return false;
}

bool ExecuteOSLInternal(const OSLShaderConfig &config,
                        const SurfaceInteraction &si, OSL::ShaderGlobals *sg,
                        OSL::ShaderGroupRef *groupRef,
                        OSL::ShadingContext **ctxOut) {
    auto &runtime = GetRuntime();
    OSL::ShaderGroupRef group = runtime.ResolveGroup(config);
    if (!group) {
        ++gCounters.executeFailures;
        return false;
    }
    OSL::ShadingContext *ctx = runtime.GetContext();
    OSLRenderState renderState;
    InitializeShaderGlobals(si, runtime.Renderer(), &renderState, sg);
    ++gCounters.executions;
    if (!runtime.System()->execute(*ctx, *group, *sg)) {
        ++gCounters.executeFailures;
        return false;
    }
    if (groupRef) *groupRef = group;
    if (ctxOut) *ctxOut = ctx;
    return true;
}

bool FindAndExtractOutput(const OSLShaderConfig &config,
                          const OSL::ShadingContext &ctx,
                          const OSL::ShaderGroup &group, bool wantSpectrum,
                          Float *floatResult, Spectrum *spectrumResult) {
    auto &runtime = GetRuntime();
    const OSL::ShaderSymbol *sym = runtime.ResolveOutputSymbol(config, group);
    if (!sym) return false;
    OSL::TypeDesc type = runtime.System()->symbol_typedesc(sym);
    const void *address = runtime.System()->symbol_address(ctx, sym);
    if (wantSpectrum) return ExtractSpectrum(type, address, spectrumResult);
    return ExtractFloat(type, address, floatResult);
}

}  // namespace
#endif

class OSLFloatTexture : public Texture<Float> {
  public:
    OSLFloatTexture(OSLShaderConfig config, Float fallbackValue)
        : config(std::move(config)), fallbackValue(fallbackValue) {}

    Float Evaluate(const SurfaceInteraction &si) const override {
        Float value = fallbackValue;
#ifdef PBRT_ENABLE_OSL
        if (EvaluateOSLFloatOutput(config, si, fallbackValue, &value)) return value;
#else
        WarnIfOSLDisabled();
#endif
        return value;
    }

  private:
    const OSLShaderConfig config;
    const Float fallbackValue;
};

class OSLSpectrumTexture : public Texture<Spectrum> {
  public:
    OSLSpectrumTexture(OSLShaderConfig config, const Spectrum &fallbackValue)
        : config(std::move(config)), fallbackValue(fallbackValue) {}

    Spectrum Evaluate(const SurfaceInteraction &si) const override {
        Spectrum value = fallbackValue;
#ifdef PBRT_ENABLE_OSL
        if (EvaluateOSLSpectrumOutput(config, si, fallbackValue, &value))
            return value;
#else
        WarnIfOSLDisabled();
#endif
        return value;
    }

  private:
    const OSLShaderConfig config;
    const Spectrum fallbackValue;
};

}  // namespace

Texture<Float> *CreateOSLFloatTexture(const Transform &tex2world,
                                      const TextureParams &tp) {
    // OSL texture contract: shader or groupspec + optional group/layer/output.
    std::string shader = tp.FindString("shader", "");
    std::string outputName = tp.FindString("output", "result");
    std::string layer = tp.FindString("layer", "");
    std::string group = tp.FindString("group", "");
    std::string groupSpec = tp.FindString("groupspec", "");
    Float fallbackValue = tp.FindFloat("default", 0.f);
    OSLShaderConfig config{shader, outputName, layer, group, groupSpec};
    ValidateAndNormalizeOSLShaderConfig(&config, "Texture \"osl\" (float)");
    return new OSLFloatTexture(config, fallbackValue);
}

Texture<Spectrum> *CreateOSLSpectrumTexture(const Transform &tex2world,
                                            const TextureParams &tp) {
    // OSL texture contract: shader or groupspec + optional group/layer/output.
    std::string shader = tp.FindString("shader", "");
    std::string outputName = tp.FindString("output", "Cs");
    std::string layer = tp.FindString("layer", "");
    std::string group = tp.FindString("group", "");
    std::string groupSpec = tp.FindString("groupspec", "");
    Spectrum fallbackValue = tp.FindSpectrum("defaultcolor", Spectrum(0.5f));
    OSLShaderConfig config{shader, outputName, layer, group, groupSpec};
    ValidateAndNormalizeOSLShaderConfig(&config, "Texture \"osl\" (spectrum)");
    return new OSLSpectrumTexture(config, fallbackValue);
}

std::shared_ptr<Texture<Float>> CreateOSLFloatTextureForOutput(
    const std::string &shader, const std::string &outputName,
    Float fallbackValue, const std::string &layer, const std::string &group,
    const std::string &groupSpec) {
    OSLShaderConfig config{shader, outputName, layer, group, groupSpec};
    return std::shared_ptr<Texture<Float>>(
        new OSLFloatTexture(config, fallbackValue));
}

std::shared_ptr<Texture<Spectrum>> CreateOSLSpectrumTextureForOutput(
    const std::string &shader, const std::string &outputName,
    const Spectrum &fallbackValue, const std::string &layer,
    const std::string &group, const std::string &groupSpec) {
    OSLShaderConfig config{shader, outputName, layer, group, groupSpec};
    return std::shared_ptr<Texture<Spectrum>>(
        new OSLSpectrumTexture(config, fallbackValue));
}

#ifdef PBRT_ENABLE_OSL
bool ExecuteOSLShader(const OSLShaderConfig &config,
                      const SurfaceInteraction &si, OSL::ShaderGlobals *sg) {
    return ExecuteOSLInternal(config, si, sg, nullptr, nullptr);
}

bool EvaluateOSLFloatOutput(const OSLShaderConfig &config,
                            const SurfaceInteraction &si, Float fallbackValue,
                            Float *result) {
    OSLShaderConfig normalized = config;
    if (!ValidateAndNormalizeOSLShaderConfig(&normalized, "OSL float evaluate")) {
        ++gCounters.fallbackCount;
        *result = fallbackValue;
        return false;
    }
    OSL::ShaderGlobals sg;
    OSL::ShaderGroupRef group;
    OSL::ShadingContext *ctx = nullptr;
    if (!ExecuteOSLInternal(normalized, si, &sg, &group, &ctx)) {
        *result = fallbackValue;
        ++gCounters.fallbackCount;
        return false;
    }

    Float value = fallbackValue;
    if (!FindAndExtractOutput(normalized, *ctx, *group, false, &value, nullptr)) {
        if (!gMissingOutputWarned.exchange(true))
            Warning("OSL output \"%s\" was not found as a float.",
                    normalized.outputName.c_str());
        *result = fallbackValue;
        ++gCounters.fallbackCount;
        return false;
    }
    *result = value;
    return true;
}

bool EvaluateOSLSpectrumOutput(const OSLShaderConfig &config,
                               const SurfaceInteraction &si,
                               const Spectrum &fallbackValue, Spectrum *result) {
    OSLShaderConfig normalized = config;
    if (!ValidateAndNormalizeOSLShaderConfig(&normalized,
                                             "OSL spectrum evaluate")) {
        ++gCounters.fallbackCount;
        *result = fallbackValue;
        return false;
    }
    OSL::ShaderGlobals sg;
    OSL::ShaderGroupRef group;
    OSL::ShadingContext *ctx = nullptr;
    if (!ExecuteOSLInternal(normalized, si, &sg, &group, &ctx)) {
        *result = fallbackValue;
        ++gCounters.fallbackCount;
        return false;
    }

    Spectrum value = fallbackValue;
    if (!FindAndExtractOutput(normalized, *ctx, *group, true, nullptr, &value)) {
        if (!gMissingOutputWarned.exchange(true))
            Warning("OSL output \"%s\" was not found as color/spectrum.",
                    normalized.outputName.c_str());
        *result = fallbackValue;
        ++gCounters.fallbackCount;
        return false;
    }
    *result = value;
    return true;
}

int GetOSLClosureIdByName(const char *name) { return GetRuntime().GetClosureId(name); }

OSLRuntimeCounters GetOSLRuntimeCounters() {
    OSLRuntimeCounters stats;
    stats.executions = gCounters.executions.load();
    stats.groupCacheHits = gCounters.groupCacheHits.load();
    stats.groupCacheMisses = gCounters.groupCacheMisses.load();
    stats.symbolCacheHits = gCounters.symbolCacheHits.load();
    stats.symbolCacheMisses = gCounters.symbolCacheMisses.load();
    stats.executeFailures = gCounters.executeFailures.load();
    stats.fallbackCount = gCounters.fallbackCount.load();
    return stats;
}
#endif

}  // namespace pbrt
