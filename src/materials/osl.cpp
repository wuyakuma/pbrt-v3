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

// materials/osl.cpp*
#include "materials/osl.h"

#include "interaction.h"
#include "microfacet.h"
#include "paramset.h"
#include "reflection.h"
#include "textures/osl.h"

#include <atomic>

namespace pbrt {

namespace {

bool HasFloatInput(const TextureParams &mp, const std::string &name) {
    int nValues = 0;
    if (mp.GetGeomParams().FindFloat(name, &nValues) && nValues > 0) return true;
    if (mp.GetMaterialParams().FindFloat(name, &nValues) && nValues > 0)
        return true;
    return mp.GetGeomParams().FindTexture(name) != "" ||
           mp.GetMaterialParams().FindTexture(name) != "";
}

bool HasSpectrumInput(const TextureParams &mp, const std::string &name) {
    int nValues = 0;
    if (mp.GetGeomParams().FindSpectrum(name, &nValues) && nValues > 0)
        return true;
    if (mp.GetMaterialParams().FindSpectrum(name, &nValues) && nValues > 0)
        return true;
    return mp.GetGeomParams().FindTexture(name) != "" ||
           mp.GetMaterialParams().FindTexture(name) != "";
}

std::atomic<bool> gMissingOSLMaterialShaderWarned(false);
std::atomic<bool> gUnknownClosureWarned(false);

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

Spectrum ToSpectrum(const OSL::Color3 &c) {
    Float rgb[3] = {Float(c.x), Float(c.y), Float(c.z)};
    return Spectrum::FromRGB(rgb);
}

void AddClosureToBSDF(const OSL::ClosureColor *closure, const OSL::Color3 &weight,
                      SurfaceInteraction *si, MemoryArena &arena,
                      TransportMode mode) {
    if (!closure) return;
    if (closure->id == OSL::ClosureColor::MUL) {
        const OSL::ClosureMul *mul = closure->as_mul();
        AddClosureToBSDF(mul->closure, weight * mul->weight, si, arena, mode);
        return;
    }
    if (closure->id == OSL::ClosureColor::ADD) {
        const OSL::ClosureAdd *add = closure->as_add();
        AddClosureToBSDF(add->closureA, weight, si, arena, mode);
        AddClosureToBSDF(add->closureB, weight, si, arena, mode);
        return;
    }

    const OSL::ClosureComponent *comp = closure->as_comp();
    Spectrum w = ToSpectrum(weight * comp->w).Clamp();
    if (w.IsBlack()) return;

    if (comp->id == DIFFUSE_ID) {
        si->bsdf->Add(ARENA_ALLOC(arena, LambertianReflection)(w));
        return;
    }
    if (comp->id == OREN_NAYAR_ID) {
        const OrenNayarParams *params = comp->as<OrenNayarParams>();
        si->bsdf->Add(ARENA_ALLOC(arena, OrenNayar)(
            w, Clamp(Float(params->sigma), Float(0), Float(90))));
        return;
    }
    if (comp->id == REFLECTION_ID || comp->id == FRESNEL_REFLECTION_ID) {
        const ReflectionParams *params = comp->as<ReflectionParams>();
        Float eta = std::max(Float(1.001f), Float(params->eta));
        si->bsdf->Add(ARENA_ALLOC(arena, SpecularReflection)(
            w, ARENA_ALLOC(arena, FresnelDielectric)(1.f, eta)));
        return;
    }
    if (comp->id == REFRACTION_ID) {
        const RefractionParams *params = comp->as<RefractionParams>();
        Float eta = std::max(Float(1.001f), Float(params->eta));
        si->bsdf->Add(ARENA_ALLOC(arena, SpecularTransmission)(w, 1.f, eta, mode));
        return;
    }
    if (comp->id == MICROFACET_ID) {
        const MicrofacetParams *params = comp->as<MicrofacetParams>();
        Float eta = std::max(Float(1.001f), Float(params->eta));
        if (params->refract == 0 || params->refract == 2) {
            MicrofacetDistribution *d = nullptr;
            if (params->dist.string() == "ggx")
                d = ARENA_ALLOC(arena, TrowbridgeReitzDistribution)(
                    std::max(Float(0.001f), Float(params->xalpha)),
                    std::max(Float(0.001f), Float(params->yalpha)));
            else
                d = ARENA_ALLOC(arena, BeckmannDistribution)(
                    std::max(Float(0.001f), Float(params->xalpha)),
                    std::max(Float(0.001f), Float(params->yalpha)));
            si->bsdf->Add(ARENA_ALLOC(arena, MicrofacetReflection)(
                w, d, ARENA_ALLOC(arena, FresnelDielectric)(1.f, eta)));
        }
        if (params->refract == 1 || params->refract == 2) {
            MicrofacetDistribution *d = nullptr;
            if (params->dist.string() == "ggx")
                d = ARENA_ALLOC(arena, TrowbridgeReitzDistribution)(
                    std::max(Float(0.001f), Float(params->xalpha)),
                    std::max(Float(0.001f), Float(params->yalpha)));
            else
                d = ARENA_ALLOC(arena, BeckmannDistribution)(
                    std::max(Float(0.001f), Float(params->xalpha)),
                    std::max(Float(0.001f), Float(params->yalpha)));
            si->bsdf->Add(ARENA_ALLOC(arena, MicrofacetTransmission)(w, d, 1.f,
                                                                      eta, mode));
        }
        return;
    }
    if (comp->id == DIELECTRIC_BSDF_ID) {
        const DielectricBsdfParams *params = comp->as<DielectricBsdfParams>();
        Float eta = std::max(Float(1.001f), Float(params->ior));
        const std::string dist = params->distribution.string();
        const Float ax = std::max(Float(0.001f), Float(params->roughness_x));
        const Float ay = std::max(Float(0.001f), Float(params->roughness_y));
        MicrofacetDistribution *d = nullptr;
        if (dist == "ggx")
            d = ARENA_ALLOC(arena, TrowbridgeReitzDistribution)(ax, ay);
        else
            d = ARENA_ALLOC(arena, BeckmannDistribution)(ax, ay);
        Spectrum reflTint = ToSpectrum(params->reflection_tint).Clamp();
        Spectrum transTint = ToSpectrum(params->transmission_tint).Clamp();
        if (!reflTint.IsBlack()) {
            si->bsdf->Add(ARENA_ALLOC(arena, MicrofacetReflection)(
                w * reflTint, d, ARENA_ALLOC(arena, FresnelDielectric)(1.f, eta)));
        }
        if (!transTint.IsBlack()) {
            MicrofacetDistribution *dt = nullptr;
            if (dist == "ggx")
                dt = ARENA_ALLOC(arena, TrowbridgeReitzDistribution)(ax, ay);
            else
                dt = ARENA_ALLOC(arena, BeckmannDistribution)(ax, ay);
            si->bsdf->Add(ARENA_ALLOC(arena, MicrofacetTransmission)(
                w * transTint, dt, 1.f, eta, mode));
        }
        return;
    }
    if (comp->id == CONDUCTOR_BSDF_ID) {
        const ConductorBsdfParams *params = comp->as<ConductorBsdfParams>();
        const std::string dist = params->distribution.string();
        const Float ax = std::max(Float(0.001f), Float(params->roughness_x));
        const Float ay = std::max(Float(0.001f), Float(params->roughness_y));
        MicrofacetDistribution *d = nullptr;
        if (dist == "ggx")
            d = ARENA_ALLOC(arena, TrowbridgeReitzDistribution)(ax, ay);
        else
            d = ARENA_ALLOC(arena, BeckmannDistribution)(ax, ay);
        Spectrum eta = ToSpectrum(params->ior).Clamp();
        Spectrum k = ToSpectrum(params->extinction).Clamp();
        si->bsdf->Add(ARENA_ALLOC(arena, MicrofacetReflection)(
            w, d, ARENA_ALLOC(arena, FresnelConductor)(Spectrum(1.f), eta, k)));
        return;
    }
    if (comp->id == TRANSPARENT_ID || comp->id == TRANSPARENT_BSDF_ID) {
        si->bsdf->Add(ARENA_ALLOC(arena, SpecularTransmission)(w, 1.f, 1.0001f,
                                                                mode));
        return;
    }

    if (!gUnknownClosureWarned.exchange(true))
        Warning("Unsupported OSL closure id %d in Material \"osl\".", comp->id);
}
#endif

}  // namespace

void OSLMaterial::ComputeScatteringFunctions(SurfaceInteraction *si,
                                             MemoryArena &arena,
                                             TransportMode mode,
                                             bool allowMultipleLobes) const {
    if (bumpMap) Bump(bumpMap, si);

    si->bsdf = ARENA_ALLOC(arena, BSDF)(*si);
#ifdef PBRT_ENABLE_OSL
    OSLShaderConfig config{shader, "Ci", layer, group, groupSpec};
    OSL::ShaderGlobals sg;
    if (ExecuteOSLShader(config, *si, &sg) && sg.Ci) {
        AddClosureToBSDF(sg.Ci, OSL::Color3(1.f), si, arena, mode);
        if (si->bsdf->NumComponents() > 0) return;
    }
#endif

    Spectrum kd = baseColor->Evaluate(*si).Clamp();
    Float sigma = Clamp(roughness->Evaluate(*si), Float(0), Float(1)) * 90.f;

    if (!kd.IsBlack()) {
        if (sigma <= 0.01f)
            si->bsdf->Add(ARENA_ALLOC(arena, LambertianReflection)(kd));
        else
            si->bsdf->Add(ARENA_ALLOC(arena, OrenNayar)(kd, sigma));
    }
}

OSLMaterial *CreateOSLMaterial(const TextureParams &mp) {
    // OSL material contract mirrors texture contract: shader/groupspec with optional group/layer.
    std::string shader = mp.FindString("shader", "");
    std::string layer = mp.FindString("layer", "");
    std::string group = mp.FindString("group", "");
    std::string groupSpec = mp.FindString("groupspec", "");

    if (shader.empty() && groupSpec.empty() &&
        !gMissingOSLMaterialShaderWarned.exchange(true)) {
        Warning(
            "Material \"osl\" expects \"shader\" or \"groupspec\". "
            "Falling back to default base color/roughness outputs.");
    }

    Spectrum defaultBaseColor = mp.FindSpectrum("default_base_color", Spectrum(0.5f));
    Float defaultRoughness = mp.FindFloat("default_roughness", 0.f);

    std::shared_ptr<Texture<Spectrum>> baseColor;
    if (HasSpectrumInput(mp, "base_color"))
        baseColor = mp.GetSpectrumTexture("base_color", defaultBaseColor);
    else
        baseColor = CreateOSLSpectrumTextureForOutput(shader, "base_color",
                                                      defaultBaseColor, layer,
                                                      group, groupSpec);

    std::shared_ptr<Texture<Float>> roughness;
    if (HasFloatInput(mp, "roughness"))
        roughness = mp.GetFloatTexture("roughness", defaultRoughness);
    else
        roughness = CreateOSLFloatTextureForOutput(shader, "roughness",
                                                   defaultRoughness, layer,
                                                   group, groupSpec);

    std::shared_ptr<Texture<Float>> bumpMap = mp.GetFloatTextureOrNull("bumpmap");
    return new OSLMaterial(shader, layer, group, groupSpec, baseColor, roughness,
                           bumpMap);
}

}  // namespace pbrt
