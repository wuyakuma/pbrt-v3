#include "sg.h"

#include "reflection.h"

namespace pbrt {

Float SGExpm1OverX(Float x) {
    if (std::abs(x) < Float(1e-4)) {
        Float x2 = x * x;
        Float x3 = x2 * x;
        Float x4 = x2 * x2;
        return 1 + x / 2 + x2 / 6 + x3 / 24 + x4 / 120;
    }
    return std::expm1(x) / x;
}

Float SGEvaluate(const Vector3f &dir, const Vector3f &axis, Float sharpness) {
    Vector3f d = dir - axis;
    return std::exp(Float(-0.5) * sharpness * d.LengthSquared());
}

Float SGIntegral(Float sharpness) {
    if (sharpness <= 0) return 4 * Pi;
    return 4 * Pi * SGExpm1OverX(-2 * sharpness);
}

SGLobe SGProduct(const Vector3f &axis1, Float sharpness1,
                 const Vector3f &axis2, Float sharpness2) {
    Vector3f axis = axis1 * sharpness1 + axis2 * sharpness2;
    Float sharpness = axis.Length();
    Vector3f d = axis1 - axis2;
    Float denom = std::max(sharpness + sharpness1 + sharpness2,
                           std::numeric_limits<Float>::min());
    Float logAmplitude = -sharpness1 * sharpness2 * d.LengthSquared() / denom;
    Vector3f normalizedAxis =
        sharpness > 0 ? axis / sharpness : Vector3f(0, 0, 1);
    return SGLobe(normalizedAxis, sharpness, logAmplitude);
}

static Float SGNormalizedHemisphericalIntegral(Float cosine, Float sharpness) {
    if (sharpness <= 0) return Float(0.5) * (Clamp(cosine, -1, 1) + 1);
    const Float A = 0.6517328826907056171791055021459;
    const Float B = 1.3418280033141287699294252888649;
    const Float C = 7.2216687798956709087860872386955;
    Float steepness =
        sharpness * std::sqrt((Float(0.5) * sharpness + A) /
                              ((sharpness + B) * sharpness + C));
    Float denom = std::erf(steepness);
    if (denom == 0) return Float(0.5) * (Clamp(cosine, -1, 1) + 1);
    return Clamp(Float(0.5) +
                     Float(0.5) *
                         (std::erf(steepness * Clamp(cosine, -1, 1)) / denom),
                 Float(0), Float(1));
}

Float VMFHemisphericalIntegral(Float cosine, Float sharpness) {
    if (sharpness <= 0) return Float(0.5);
    Float lerpFactor = SGNormalizedHemisphericalIntegral(cosine, sharpness);
    Float e = std::exp(-sharpness);
    return Lerp(lerpFactor, e, Float(1)) / (e + 1);
}

static Float UpperSGClampedCosineIntegralOverTwoPi(Float sharpness) {
    if (sharpness <= Float(0.5)) {
        return (((((((-Float(1) / 362880) * sharpness + Float(1) / 40320) *
                        sharpness -
                    Float(1) / 5040) *
                       sharpness +
                   Float(1) / 720) *
                      sharpness -
                  Float(1) / 120) *
                     sharpness +
                 Float(1) / 24) *
                    sharpness -
                Float(1) / 6) *
                   sharpness +
               Float(0.5);
    }
    return (1 - SGExpm1OverX(-sharpness)) / sharpness;
}

static Float LowerSGClampedCosineIntegralOverTwoPi(Float sharpness) {
    Float e = std::exp(-sharpness);
    if (sharpness <= Float(0.5)) {
        return e * (((((((((Float(1) / 403200) * sharpness -
                           Float(1) / 45360) *
                              sharpness +
                          Float(1) / 5760) *
                             sharpness -
                         Float(1) / 840) *
                            sharpness +
                        Float(1) / 144) *
                           sharpness -
                       Float(1) / 30) *
                          sharpness +
                      Float(1) / 8) *
                         sharpness -
                     Float(1) / 3) *
                        sharpness +
                    Float(0.5));
    }
    return e * (SGExpm1OverX(-sharpness) - e) / sharpness;
}

Float SGClampedCosineProductIntegralOverPi2024(Float cosine, Float sharpness) {
    if (sharpness <= 0) return std::max(Float(0), cosine);

    const Float A = 2.7360831611272558028247203765204;
    const Float B = 17.02129778174187535455530451145;
    const Float C = 4.0100826728510421403939290030394;
    const Float D = 15.219156263147210594866010069381;
    const Float E = 76.087896272360737270901154261082;
    Float t = sharpness *
              std::sqrt(Float(0.5) * ((sharpness + A) * sharpness + B) /
                        (((sharpness + C) * sharpness + D) * sharpness + E));
    Float tz = t * Clamp(cosine, -1, 1);
    const Float invSqrtPi = 0.56418958354775628694807945156077;
    Float lerpFactor =
        Float(0.5) * (cosine * std::erfc(-tz) + std::erfc(t)) -
        Float(0.5) * invSqrtPi * std::exp(-tz * tz) *
            std::expm1(t * t * (cosine * cosine - 1)) /
            std::max(t, Float(MachineEpsilon));
    lerpFactor = Clamp(lerpFactor, Float(0), Float(1));

    Float lowerIntegral = LowerSGClampedCosineIntegralOverTwoPi(sharpness);
    Float upperIntegral = UpperSGClampedCosineIntegralOverTwoPi(sharpness);
    return 2 * Lerp(lerpFactor, lowerIntegral, upperIntegral);
}

Float VMFAxisLengthToSharpness(Float axisLength) {
    axisLength =
        Clamp(axisLength, Float(0), Float(1) - Float(MachineEpsilon));
    return axisLength * (3 - axisLength * axisLength) /
           std::max(Float(1) - axisLength * axisLength, Float(MachineEpsilon));
}

Float VMFSharpnessToAxisLength(Float sharpness) {
    if (sharpness <= 0) return 0;
    Float a = sharpness / 3;
    Float b = a * a * a;
    Float c = std::sqrt(1 + 3 * a * a * (1 + a * a));
    Float theta = std::atan2(c, b) / 3;
    Float d = -2 * std::sin(Pi / 6 - theta);
    return sharpness > Float(33554432) ? Float(1)
                                       : std::sqrt(1 + a * a) * d + a;
}

Vector3f GGXDominantVisibleNormal(const Vector3f &wi, Float alphax,
                                  Float alphay) {
    Vector2f v(alphax * wi.x, alphay * wi.y);
    Float len2 = v.LengthSquared();
    Float t = std::sqrt(len2 + wi.z * wi.z);
    Float z = wi.z >= 0 ? t + wi.z
                         : len2 / std::max(t - wi.z, Float(MachineEpsilon));
    return Normalize(Vector3f(alphax * alphax * wi.x, alphay * alphay * wi.y, z));
}

static Float SGGX(const Vector3f &m, const SGMatrix2x2 &roughnessMat) {
    Float det = std::max(SGMatrixDeterminant(roughnessMat),
                         Float(MachineEpsilon));
    SGMatrix2x2 adj(roughnessMat.m11, -roughnessMat.m01, -roughnessMat.m10,
                    roughnessMat.m00);
    Vector2f mxy(m.x, m.y);
    Float length2 = Dot(mxy, SGMatrixMul(adj, mxy)) / det + m.z * m.z;
    return 1 / (Pi * std::sqrt(det) * length2 * length2);
}

Float SGGXReflectionPDF(const Vector3f &wi, const Vector3f &m,
                        const SGMatrix2x2 &roughnessMat) {
    Vector2f wiXY(wi.x, wi.y);
    Float denom = 4 * std::sqrt(std::max(
                            Dot(wiXY, SGMatrixMul(roughnessMat, wiXY)) +
                                wi.z * wi.z,
                            Float(MachineEpsilon)));
    return SGGX(m, roughnessMat) / denom;
}

}  // namespace pbrt
