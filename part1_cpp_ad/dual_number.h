#ifndef DUAL_NUMBER_H
#define DUAL_NUMBER_H

#include <cmath>
#include <vector>
#include <iostream>

class dual_number {
public:
    float val;
    float dual;

    dual_number() : val(0.0f), dual(0.0f) {}
    dual_number(float v) : val(v), dual(0.0f) {}
    dual_number(float v, float d) : val(v), dual(d) {}

    float value() const { return val; }
    float derivative() const { return dual; }

    dual_number operator+(const dual_number& rhs) const {
        return dual_number(val + rhs.val, dual + rhs.dual);
    }

    dual_number operator-(const dual_number& rhs) const {
        return dual_number(val - rhs.val, dual - rhs.dual);
    }

    dual_number operator*(const dual_number& rhs) const {
        return dual_number(val * rhs.val, val * rhs.dual + dual * rhs.val);
    }

    dual_number operator/(const dual_number& rhs) const {
        float denom = rhs.val * rhs.val;
        return dual_number(val / rhs.val, (dual * rhs.val - val * rhs.dual) / denom);
    }

    friend dual_number sin(const dual_number& x) {
        return dual_number(std::sin(x.val), std::cos(x.val) * x.dual);
    }

    friend dual_number cos(const dual_number& x) {
        return dual_number(std::cos(x.val), -std::sin(x.val) * x.dual);
    }

    friend dual_number exp(const dual_number& x) {
        float e = std::exp(x.val);
        return dual_number(e, e * x.dual);
    }

    friend dual_number ln(const dual_number& x) {
        return dual_number(std::log(x.val), x.dual / x.val);
    }

    friend dual_number relu(const dual_number& x) {
        return dual_number(x.val > 0 ? x.val : 0, x.val > 0 ? x.dual : 0);
    }

    friend dual_number sigmoid(const dual_number& x) {
        float s = 1 / (1 + std::exp(-x.val));
        return dual_number(s, s * (1 - s) * x.dual);
    }

    friend dual_number tanh(const dual_number& x) {
        float t = std::tanh(x.val);
        return dual_number(t, (1 - t * t) * x.dual);
    }
};

class dual_vector {
public:
    std::vector<dual_number> data;

    dual_vector(size_t n) : data(n) {}
    dual_vector(std::initializer_list<dual_number> init) : data(init) {}

    size_t size() const { return data.size(); }

    dual_number& operator[](size_t i) { return data[i]; }
    const dual_number& operator[](size_t i) const { return data[i]; }

    dual_vector apply(dual_number (*func)(const dual_number&)) const {
        dual_vector result(data.size());
        for (size_t i = 0; i < data.size(); ++i)
            result[i] = func(data[i]);
        return result;
    }
};

#endif // DUAL_NUMBER_H
