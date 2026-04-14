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

#include <atomic>
#include <mutex>
#include <unordered_map>

#ifdef PBRT_ENABLE_OSL
#include <OSL/genclosure.h>
#include <OSL/rendererservices.h>
#endif

namespace pbrt {

namespace {

std::atomic<bool> gMissingShaderWarned(false);
std::atomic<bool> gMissingOutputWarned(false);
std::atomic<bool> gOSLDisabledWarned(false);

void WarnIfMissingShader(const OSLShaderConfig &config) {
    if (!config.shader.empty() || !config.groupSpec.empty()) return;
    if (!gMissingShaderWarned.exchange(true))
        Warning("OSL entry requires \"shader\" or \"groupspec\".");
}

#ifndef PBRT_ENABLE_OSL
void WarnIfOSLDisabled() {
    if (!gOSLDisabledWarned.exchange(true))
        Warning("OSL support is disabled. Reconfigure with PBRT_ENABLE_OSL=ON.");
}
#endif

#ifdef PBRT_ENABLE_OSL

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

class PbrtOSLRendererServices : public OSL::RendererServices {
  public:
    PbrtOSLRendererServices() : OSL::RendererServices(nullptr) {}

    bool get_matrix(OSL::ShaderGlobals *sg, OSL::Matrix44 &result,
                    OSL::TransformationPtr xform, float time) override {
        result.makeIdentity();
        return true;
    }

    bool get_matrix(OSL::ShaderGlobals *sg, OSL::Matrix44 &result,
                    OSL::TransformationPtr xform) override {
        result.makeIdentity();
        return true;
    }

    bool get_matrix(OSL::ShaderGlobals *sg, OSL::Matrix44 &result,
                    OSL::ustringhash from, float time) override {
        result.makeIdentity();
        return true;
    }

    bool get_matrix(OSL::ShaderGlobals *sg, OSL::Matrix44 &result,
                    OSL::ustringhash from) override {
        result.makeIdentity();
        return true;
    }

    bool get_inverse_matrix(OSL::ShaderGlobals *sg, OSL::Matrix44 &result,
                            OSL::TransformationPtr xform,
                            float time) override {
        result.makeIdentity();
        return true;
    }

    bool get_inverse_matrix(OSL::ShaderGlobals *sg, OSL::Matrix44 &result,
                            OSL::TransformationPtr xform) override {
        result.makeIdentity();
        return true;
    }

    bool get_inverse_matrix(OSL::ShaderGlobals *sg, OSL::Matrix44 &result,
                            OSL::ustringhash to, float time) override {
        result.makeIdentity();
        return true;
    }

    bool get_inverse_matrix(OSL::ShaderGlobals *sg, OSL::Matrix44 &result,
                            OSL::ustringhash to) override {
        result.makeIdentity();
        return true;
    }
};

struct OSLRuntime {
    OSLRuntime() : renderer(new PbrtOSLRendererServices()), ss(new OSL::ShadingSystem(renderer.get(), nullptr, nullptr)) {
        RegisterClosures();
    }

    OSL::ShadingContext *GetContext() {
        if (!tlsThreadInfo) tlsThreadInfo = ss->create_thread_info();
        if (!tlsShadingContext) tlsShadingContext = ss->get_context(tlsThreadInfo);
        return tlsShadingContext;
    }

    OSL::ShaderGroupRef ResolveGroup(const OSLShaderConfig &config) {
        const std::string key =
            config.shader + "|" + config.group + "|" + config.layer + "|" +
            config.groupSpec + "|" + config.outputName;
        std::lock_guard<std::mutex> lock(cacheMutex);
        auto iter = groups.find(key);
        if (iter != groups.end()) return iter->second;

        const std::string groupName =
            config.group.empty() ? (config.shader.empty() ? "pbrt_osl_group" : config.shader)
                                 : config.group;
        OSL::ShaderGroupRef groupRef;
        if (!config.groupSpec.empty()) {
            groupRef = ss->ShaderGroupBegin(groupName, "surface", config.groupSpec);
            if (!groupRef) return OSL::ShaderGroupRef();
            ss->ShaderGroupEnd(*groupRef);
        } else {
            if (config.shader.empty()) return OSL::ShaderGroupRef();
            groupRef = ss->ShaderGroupBegin(groupName);
            if (!groupRef) return OSL::ShaderGroupRef();
            const std::string layerName =
                config.layer.empty() ? "pbrt_osl_layer" : config.layer;
            if (!ss->Shader(*groupRef, "surface", config.shader, layerName)) {
                return OSL::ShaderGroupRef();
            }
            ss->ShaderGroupEnd(*groupRef);
        }

        std::vector<const char *> outputs;
        outputs.push_back("Ci");
        if (!config.outputName.empty()) outputs.push_back(config.outputName.c_str());
        ss->attribute(groupRef.get(), "renderer_outputs",
                      OSL::TypeDesc(OSL::TypeDesc::STRING, int(outputs.size())),
                      outputs.data());
        ss->optimize_group(groupRef.get(), GetContext(), true);
        groups.emplace(key, groupRef);
        return groupRef;
    }

    int GetClosureId(const char *name) {
        if (!name) return -1;
        int id = -1;
        const char *queryName = name;
        const OSL::ClosureParam *params = nullptr;
        if (ss->query_closure(&queryName, &id, &params)) return id;
        return -1;
    }

    std::unique_ptr<PbrtOSLRendererServices> renderer;
    std::unique_ptr<OSL::ShadingSystem> ss;
    std::mutex cacheMutex;
    std::unordered_map<std::string, OSL::ShaderGroupRef> groups;

    static thread_local OSL::PerThreadInfo *tlsThreadInfo;
    static thread_local OSL::ShadingContext *tlsShadingContext;

  private:
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

thread_local OSL::PerThreadInfo *OSLRuntime::tlsThreadInfo = nullptr;
thread_local OSL::ShadingContext *OSLRuntime::tlsShadingContext = nullptr;

OSLRuntime &GetRuntime() {
    static OSLRuntime runtime;
    return runtime;
}

OSL::Vec3 ToOSL(const Vector3f &v) { return OSL::Vec3(v.x, v.y, v.z); }
OSL::Vec3 ToOSL(const Point3f &p) { return OSL::Vec3(p.x, p.y, p.z); }

Spectrum ToSpectrum(const OSL::Vec3 &v) {
    Float rgb[3] = {Float(v.x), Float(v.y), Float(v.z)};
    return Spectrum::FromRGB(rgb);
}

void InitializeShaderGlobals(const SurfaceInteraction &si,
                             OSL::RendererServices *renderer,
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
    WarnIfMissingShader(config);
    auto &runtime = GetRuntime();
    OSL::ShaderGroupRef group = runtime.ResolveGroup(config);
    if (!group) return false;
    OSL::ShadingContext *ctx = runtime.GetContext();
    InitializeShaderGlobals(si, runtime.renderer.get(), sg);
    if (!runtime.ss->execute(*ctx, *group, *sg)) return false;
    if (groupRef) *groupRef = group;
    if (ctxOut) *ctxOut = ctx;
    return true;
}

bool FindAndExtractOutput(const OSLShaderConfig &config,
                          const OSL::ShadingContext &ctx,
                          const OSL::ShaderGroup &group, bool wantSpectrum,
                          Float *floatResult, Spectrum *spectrumResult) {
    auto &runtime = GetRuntime();
    const OSL::ShaderSymbol *sym = nullptr;
    if (!config.layer.empty())
        sym = runtime.ss->find_symbol(group, OSL::ustring(config.layer),
                                      OSL::ustring(config.outputName));
    if (!sym)
        sym = runtime.ss->find_symbol(group, OSL::ustring(config.outputName));
    if (!sym) return false;
    OSL::TypeDesc type = runtime.ss->symbol_typedesc(sym);
    const void *address = runtime.ss->symbol_address(ctx, sym);
    if (wantSpectrum) return ExtractSpectrum(type, address, spectrumResult);
    return ExtractFloat(type, address, floatResult);
}
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
    OSL::ShaderGlobals sg;
    OSL::ShaderGroupRef group;
    OSL::ShadingContext *ctx = nullptr;
    if (!ExecuteOSLInternal(config, si, &sg, &group, &ctx)) {
        *result = fallbackValue;
        return false;
    }

    Float value = fallbackValue;
    if (!FindAndExtractOutput(config, *ctx, *group, false, &value, nullptr)) {
        if (!gMissingOutputWarned.exchange(true))
            Warning("OSL output \"%s\" was not found as a float.",
                    config.outputName.c_str());
        *result = fallbackValue;
        return false;
    }
    *result = value;
    return true;
}

bool EvaluateOSLSpectrumOutput(const OSLShaderConfig &config,
                               const SurfaceInteraction &si,
                               const Spectrum &fallbackValue, Spectrum *result) {
    OSL::ShaderGlobals sg;
    OSL::ShaderGroupRef group;
    OSL::ShadingContext *ctx = nullptr;
    if (!ExecuteOSLInternal(config, si, &sg, &group, &ctx)) {
        *result = fallbackValue;
        return false;
    }

    Spectrum value = fallbackValue;
    if (!FindAndExtractOutput(config, *ctx, *group, true, nullptr, &value)) {
        if (!gMissingOutputWarned.exchange(true))
            Warning("OSL output \"%s\" was not found as color/spectrum.",
                    config.outputName.c_str());
        *result = fallbackValue;
        return false;
    }
    *result = value;
    return true;
}

int GetOSLClosureIdByName(const char *name) { return GetRuntime().GetClosureId(name); }
#endif

}  // namespace pbrt
