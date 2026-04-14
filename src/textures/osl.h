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

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_TEXTURES_OSL_H
#define PBRT_TEXTURES_OSL_H

// textures/osl.h*
#include "paramset.h"
#include "pbrt.h"
#include "texture.h"

#ifdef PBRT_ENABLE_OSL
#include <OSL/oslclosure.h>
#include <OSL/oslexec.h>
#endif

namespace pbrt {

struct OSLShaderConfig {
    std::string shader;
    std::string outputName;
    std::string layer;
    std::string group;
    std::string groupSpec;
};

struct OSLRuntimeCounters {
    uint64_t executions = 0;
    uint64_t groupCacheHits = 0;
    uint64_t groupCacheMisses = 0;
    uint64_t symbolCacheHits = 0;
    uint64_t symbolCacheMisses = 0;
    uint64_t executeFailures = 0;
    uint64_t fallbackCount = 0;
};

Texture<Float> *CreateOSLFloatTexture(const Transform &tex2world,
                                      const TextureParams &tp);
Texture<Spectrum> *CreateOSLSpectrumTexture(const Transform &tex2world,
                                            const TextureParams &tp);

std::shared_ptr<Texture<Float>> CreateOSLFloatTextureForOutput(
    const std::string &shader, const std::string &outputName,
    Float fallbackValue, const std::string &layer = "",
    const std::string &group = "", const std::string &groupSpec = "");
std::shared_ptr<Texture<Spectrum>> CreateOSLSpectrumTextureForOutput(
    const std::string &shader, const std::string &outputName,
    const Spectrum &fallbackValue, const std::string &layer = "",
    const std::string &group = "", const std::string &groupSpec = "");

bool ValidateAndNormalizeOSLShaderConfig(OSLShaderConfig *config,
                                         const char *ownerLabel);

#ifdef PBRT_ENABLE_OSL
bool ExecuteOSLShader(const OSLShaderConfig &config,
                      const SurfaceInteraction &si, OSL::ShaderGlobals *sg);
bool EvaluateOSLFloatOutput(const OSLShaderConfig &config,
                            const SurfaceInteraction &si, Float fallbackValue,
                            Float *result);
bool EvaluateOSLSpectrumOutput(const OSLShaderConfig &config,
                               const SurfaceInteraction &si,
                               const Spectrum &fallbackValue, Spectrum *result);
int GetOSLClosureIdByName(const char *name);
OSLRuntimeCounters GetOSLRuntimeCounters();
#endif

}  // namespace pbrt

#endif  // PBRT_TEXTURES_OSL_H
