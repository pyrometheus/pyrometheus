#include "pyro.hxx"

#include <vector>
#include <iostream>
#include <cstddef>

#define DEF_VECTOR_OP(op) \
    Vector<T>& operator op##=(const Vector<T>& other) { \
        for (size_t i = 0; i < data.size(); ++i) { \
            data[i] op##= other[i]; \
        } \
        return *this; \
    }\
    template <typename U> \
    Vector<T>& operator op##=(U scalar) { \
        for (size_t i = 0; i < data.size(); ++i) { \
            data[i] op##= scalar; \
        } \
        return *this; \
    }

#define DEF_VECTOR_OP_OUTER(op) \
    template <typename T> \
    Vector<T> operator op(const Vector<T>& a, const Vector<T>& b) { \
        Vector<T> result = a; \
        result op##= b; \
        return result; \
    } \
    template <typename T, typename U> \
    Vector<T> operator op(const Vector<T>& a, U scalar) { \
        Vector<T> result = a; \
        result op##= scalar; \
        return result; \
    } \
    template <typename T, typename U> \
    Vector<T> operator op(U scalar, const Vector<T>& a) { \
        Vector<T> result = a; \
        result op##= scalar; \
        return result; \
    }

template <typename T>
struct Vector {
    std::vector<T> data;

    T& operator[](size_t i) { return data[i]; }
    const T& operator[](size_t i) const { return data[i]; }

    DEF_VECTOR_OP(+)
    DEF_VECTOR_OP(-)
    DEF_VECTOR_OP(*)
    DEF_VECTOR_OP(/)
};

DEF_VECTOR_OP_OUTER(+)
DEF_VECTOR_OP_OUTER(-)
DEF_VECTOR_OP_OUTER(*)
DEF_VECTOR_OP_OUTER(/)

template <typename T>
Vector<T> sqrt(const Vector<T>& a) {
    Vector<T> result = a;
    for (size_t i = 0; i < a.data.size(); ++i) {
        result[i] = sqrt(a[i]);
    }
    return result;
}

template <typename T>
Vector<T> exp(const Vector<T>& a) {
    Vector<T> result = a;
    for (size_t i = 0; i < a.data.size(); ++i) {
        result[i] = exp(a[i]);
    }
    return result;
}

template <typename T>
Vector<T> log(const Vector<T>& a) {
    Vector<T> result = a;
    for (size_t i = 0; i < a.data.size(); ++i) {
        result[i] = log(a[i]);
    }
    return result;
}

template <typename T, typename U>
Vector<T> pow(const Vector<T>& a, const U& b) {
    Vector<T> result = a;
    for (size_t i = 0; i < a.data.size(); ++i) {
        result[i] = pow(a[i], b);
    }
    return result;
}

#define DEF_SCALAR_OP(op) \
    Scalar& operator op##=(const Scalar& other) { \
        val = "(" + val + ")" + #op + "(" + other.val + ")"; \
        return *this; \
    }\
    template <typename T> \
    Scalar& operator op##=(T scalar) { \
        val = "(" + val + ")" + #op + "(" + std::to_string(scalar) + ")"; \
        return *this; \
    }

#define DEF_SCALAR_OP_OUTER(op) \
    Scalar operator op(const Scalar& a, const Scalar& b) { \
        return "(" + a.val + ")" + #op + "(" + b.val + ")"; \
    }

struct Scalar {
    std::string val;  

    Scalar() = default;
    Scalar(const std::string& other) : val(other) {}
    template <typename T>
    Scalar(const T& other) : val(std::to_string(other)) {}

    operator std::string() const { return val; }
    operator std::string&() { return val; }

    template <typename T>
    Scalar& operator=(const T& other) {
        val = std::to_string(other);
        return *this;
    }

    DEF_SCALAR_OP(+)
    DEF_SCALAR_OP(-)
    DEF_SCALAR_OP(*)
    DEF_SCALAR_OP(/)

    Scalar operator-() const { return "(-" + val + ")"; }
    Scalar operator+() const { return val; }

    // These are incorrect, but are unused for this example.
    operator double() const { return 2.0; }
    bool operator==(const Scalar& other) const { return val == other.val; }
    bool operator!=(const Scalar& other) const { return val != other.val; }
    bool operator<(const Scalar& other) const { return val < other.val; }
    bool operator<=(const Scalar& other) const { return val <= other.val; }
    bool operator>(const Scalar& other) const { return val > other.val; }
    bool operator>=(const Scalar& other) const { return val >= other.val; }
};

DEF_SCALAR_OP_OUTER(+)
DEF_SCALAR_OP_OUTER(-)
DEF_SCALAR_OP_OUTER(*)
DEF_SCALAR_OP_OUTER(/)

Scalar sqrt(const Scalar& a) {
    return "sqrt(" + a.val + ")";
}

Scalar exp(const Scalar& a) {
    return std::string("exp(") + a.val + ")";
}

Scalar log(const Scalar& a) {
    return std::string("log(") + a.val + ")";
}

template <typename U>
Scalar pow(const Scalar& a, const U& b) {
    return std::string("pow(") + a.val + ", " + std::to_string(b) + ")";
}

using mech_t = pyro::pyro<Scalar, Vector<Scalar>>;

int main(int argc, char** argv) {
    Vector<Scalar> temperatures;
    temperatures.data = {300, 300, 400};

    mech_t::Species2T diffs = mech_t::get_species_binary_mass_diffusivities(temperatures);
    for (size_t i = 0; i < diffs.size(); ++i) {
        for (size_t j = 0; j < diffs[i].size(); ++j) {
            std::cout << "i, j: " << i << ", " << j << std::endl;
            for (size_t k = 0; k < diffs[i][j].data.size(); ++k) {
                std::cout << diffs[i][j][k].val << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    return 0;
}