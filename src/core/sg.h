#ifndef PBRT_CORE_SG_H
#define PBRT_CORE_SG_H

#include "geometry.h"
#include "pbrt.h"

namespace pbrt {

struct SGLobe {
    SGLobe() : axis(0, 0, 1), sharpness(0), logAmplitude(0) {}
    SGLobe(const Vector3f &axis, Float sharpness, Float logAmplitude)
        : axis(axis), sharpness(sharpness), logAmplitude(logAmplitude) {}

    Vector3f axis;
    Float sharpness;
    Float logAmplitude;
};

struct SGMatrix2x2 {
    SGMatrix2x2() : m00(0), m01(0), m10(0), m11(0) {}
    SGMatrix2x2(Float m00, Float m01, Float m10, Float m11)
        : m00(m00), m01(m01), m10(m10), m11(m11) {}

    Float m00, m01, m10, m11;
};

Float SGExpm1OverX(Float x);
Float SGEvaluate(const Vector3f &dir, const Vector3f &axis, Float sharpness);
Float SGIntegral(Float sharpness);
SGLobe SGProduct(const Vector3f &axis1, Float sharpness1,
                 const Vector3f &axis2, Float sharpness2);
Float VMFHemisphericalIntegral(Float cosine, Float sharpness);
Float SGClampedCosineProductIntegralOverPi2024(Float cosine, Float sharpness);
Float VMFAxisLengthToSharpness(Float axisLength);
Float VMFSharpnessToAxisLength(Float sharpness);
Vector3f GGXDominantVisibleNormal(const Vector3f &wi, Float alphax,
                                  Float alphay);
Float SGGXReflectionPDF(const Vector3f &wi, const Vector3f &m,
                        const SGMatrix2x2 &roughnessMat);

inline Float SGMatrixDeterminant(const SGMatrix2x2 &m) {
    return m.m00 * m.m11 - m.m01 * m.m10;
}

inline Vector2f SGMatrixMul(const SGMatrix2x2 &m, const Vector2f &v) {
    return Vector2f(m.m00 * v.x + m.m01 * v.y,
                    m.m10 * v.x + m.m11 * v.y);
}

}  // namespace pbrt

#endif  // PBRT_CORE_SG_H
