#include "dual_number.h"
#include <gtest/gtest.h>

dual_number test_func(const dual_number& x) {
    return x * x + sin(x);
}

TEST(DualNumberTest, BasicValueAndDerivative) {
    dual_number x(1.0f, 1.0f);
    dual_number y = test_func(x);

    float expected_val = 1.0f * 1.0f + std::sin(1.0f);
    float expected_deriv = 2.0f * 1.0f + std::cos(1.0f);

    EXPECT_NEAR(y.value(), expected_val, 1e-5);
    EXPECT_NEAR(y.derivative(), expected_deriv, 1e-5);
}

TEST(DualVectorTest, ElementwiseApply) {
    dual_vector vec = {dual_number(1.0f, 1.0f), dual_number(2.0f, 1.0f), dual_number(3.0f, 1.0f)};
    auto results = vec.apply([](const dual_number& x) {
        return x * x + exp(x);
    });

    for (size_t i = 0; i < vec.size(); ++i) {
        float x_val = vec[i].value();
        float expected_val = x_val * x_val + std::exp(x_val);
        float expected_deriv = 2 * x_val + std::exp(x_val);

        EXPECT_NEAR(results[i].value(), expected_val, 1e-5);
        EXPECT_NEAR(results[i].derivative(), expected_deriv, 1e-5);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
