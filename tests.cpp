#include "gtest/gtest.h"

#include "main.h"

TEST(Test, TestFactorial5)
{
  EXPECT_EQ(factorial(5), 120);
}

TEST(Test, TestFactorial10)
{
  EXPECT_EQ(factorial(10), 3628800);
}
//////////////////////////

/////////////////////////////