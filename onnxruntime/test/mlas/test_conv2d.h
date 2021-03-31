#pragma once

#include "test_util.h"
#include "gtest/gtest.h"

class MlasConv2DTest : public MlasTestBase {
 protected:
  virtual void MlasConv2D(
      size_t BatchCount,
      size_t GroupCount,
      size_t InputChannels,
      size_t InputHeight,
      size_t InputWidth,
      size_t FilterCount,
      size_t KernelHeight,
      size_t KernelWidth,
      size_t PaddingLeftHeight,
      size_t PaddingLeftWidth,
      size_t PaddingRightHeight,
      size_t PaddingRightWidth,
      size_t DilationHeight,
      size_t DilationWidth,
      size_t StrideHeight,
      size_t StrideWidth,
      size_t OutputHeight,
      size_t OutputWidth,
      const float* Input,
      const float* Filter,
      const float* Bias,
      float* Output);

  void ReferenceConv2D(
      size_t BatchCount,
      size_t GroupCount,
      size_t InputChannels,
      size_t InputHeight,
      size_t InputWidth,
      size_t FilterCount,
      size_t KernelHeight,
      size_t KernelWidth,
      size_t PaddingLeftHeight,
      size_t PaddingLeftWidth,
      size_t DilationHeight,
      size_t DilationWidth,
      size_t StrideHeight,
      size_t StrideWidth,
      size_t OutputHeight,
      size_t OutputWidth,
      const float* Input,
      const float* Filter,
      const float* Bias,
      float* Output);

  MatrixGuardBuffer<float> BufferInput;
  MatrixGuardBuffer<float> BufferFilter;
  MatrixGuardBuffer<float> BufferBias;
  MatrixGuardBuffer<float> BufferOutput;
  MatrixGuardBuffer<float> BufferOutputReference;
  MatrixGuardBuffer<float> BufferWorking;
  MatrixGuardBuffer<float> BufferIm2Col;

  MLAS_THREADPOOL* threadpool_;

 public:
  MlasConv2DTest(MLAS_THREADPOOL* threadpool) : threadpool_(threadpool) {}

  void Test(
      size_t BatchCount,
      size_t GroupCount,
      size_t InputChannels,
      size_t InputHeight,
      size_t InputWidth,
      size_t FilterCount,
      size_t KernelHeight,
      size_t KernelWidth,
      size_t PaddingLeftHeight,
      size_t PaddingLeftWidth,
      size_t PaddingRightHeight,
      size_t PaddingRightWidth,
      size_t DilationHeight,
      size_t DilationWidth,
      size_t StrideHeight,
      size_t StrideWidth);

  void ExecuteLong(void) override;

  static const char* GetSuitePrefix() {
    return "Conv2dTest";
  }
};

class MlasNchwcConv2DTest : public MlasConv2DTest {
 protected:
  void MlasConv2D(
      size_t BatchCount,
      size_t GroupCount,
      size_t InputChannels,
      size_t InputHeight,
      size_t InputWidth,
      size_t FilterCount,
      size_t KernelHeight,
      size_t KernelWidth,
      size_t PaddingLeftHeight,
      size_t PaddingLeftWidth,
      size_t PaddingRightHeight,
      size_t PaddingRightWidth,
      size_t DilationHeight,
      size_t DilationWidth,
      size_t StrideHeight,
      size_t StrideWidth,
      size_t OutputHeight,
      size_t OutputWidth,
      const float* Input,
      const float* Filter,
      const float* Bias,
      float* Output) override;

  const size_t BlockSize = MlasNchwcGetBlockSize();

  MatrixGuardBuffer<float> BufferNchwcInput;
  MatrixGuardBuffer<float> BufferNchwcFilter;
  MatrixGuardBuffer<float> BufferNchwcBias;
  MatrixGuardBuffer<float> BufferNchwcOutput;

 public:
  MlasNchwcConv2DTest(MLAS_THREADPOOL* threadpool = nullptr) : MlasConv2DTest(threadpool) {}

  void ExecuteLong(void) override;

  static const char* GetSuitePrefix(void) {
    return "Conv2dNchwcTest";
  }
};

// Test Suite class for various conv2d test.
template <typename TConv2DTester, bool ShortExec, bool Threaded>
class Conv2dTestFixture : public testing::Test {
 public:
  static void SetUpTestSuite() {
    conv2d_tester = new TConv2DTester(Threaded ? GetMlasThreadPool() : nullptr);
  };

  static void TearDownTestSuite() {
    if (conv2d_tester != nullptr) {
      delete conv2d_tester;
    }
    conv2d_tester = nullptr;
  };

  static TConv2DTester* conv2d_tester;
};

// Short Execute treat each test seperately by all parameters.
template <typename TConv2DTester, bool Threaded>
class Conv2dShortExecuteTest : public Conv2dTestFixture<TConv2DTester, true, Threaded> {
 public:
  explicit Conv2dShortExecuteTest(
      size_t BatchCount,
      size_t GroupCount,
      size_t InputChannels,
      size_t InputHeight,
      size_t InputWidth,
      size_t FilterCount,
      size_t KernelHeight,
      size_t KernelWidth,
      size_t PaddingLeftHeight,
      size_t PaddingLeftWidth,
      size_t PaddingRightHeight,
      size_t PaddingRightWidth,
      size_t DilationHeight,
      size_t DilationWidth,
      size_t StrideHeight,
      size_t StrideWidth) : BatchCount_(BatchCount),
                            GroupCount_(GroupCount),
                            InputChannels_(InputChannels),
                            InputHeight_(InputHeight),
                            InputWidth_(InputWidth),
                            FilterCount_(FilterCount),
                            KernelHeight_(KernelHeight),
                            KernelWidth_(KernelWidth),
                            PaddingLeftHeight_(PaddingLeftHeight),
                            PaddingLeftWidth_(PaddingLeftWidth),
                            PaddingRightHeight_(PaddingRightHeight),
                            PaddingRightWidth_(PaddingRightWidth),
                            DilationHeight_(DilationHeight),
                            DilationWidth_(DilationWidth),
                            StrideHeight_(StrideHeight),
                            StrideWidth_(StrideWidth) {
  }

  void TestBody() override {
    conv2d_tester->Test(
        BatchCount_,
        GroupCount_,
        InputChannels_,
        InputHeight_,
        InputWidth_,
        FilterCount_,
        KernelHeight_,
        KernelWidth_,
        PaddingLeftHeight_,
        PaddingLeftWidth_,
        PaddingRightHeight_,
        PaddingRightWidth_,
        DilationHeight_,
        DilationWidth_,
        StrideHeight_,
        StrideWidth_);
  }

  static size_t RegisterSingleTest(
      size_t BatchCount,
      size_t GroupCount,
      size_t InputChannels,
      size_t InputHeight,
      size_t InputWidth,
      size_t FilterCount,
      size_t KernelHeight,
      size_t KernelWidth,
      size_t PaddingLeftHeight,
      size_t PaddingLeftWidth,
      size_t PaddingRightHeight,
      size_t PaddingRightWidth,
      size_t DilationHeight,
      size_t DilationWidth,
      size_t StrideHeight,
      size_t StrideWidth) {
    static const std::string suite_name =
        std::string(TConv2DTester::GetSuitePrefix()) + (Threaded ? "_Short_Threaded" : "_Short_SingleThread");

    std::stringstream ss;
    ss << "B" << BatchCount << "/"
       << "G" << GroupCount << "/"
       << "Cpg" << InputChannels << "/"
       << "Fpg" << FilterCount << "/"
       << "H" << InputHeight << "/"
       << "W" << InputWidth << "/"
       << "KH" << KernelHeight << "/"
       << "KW" << KernelWidth << "/"
       << "Pad" << PaddingLeftHeight << "," << PaddingLeftWidth << "," << PaddingRightHeight << "," << PaddingRightWidth << "/"
       << "Dilation" << DilationHeight << "," << DilationWidth << "/"
       << "Stride" << StrideHeight << "," << StrideWidth;
    auto test_name = ss.str();

    testing::RegisterTest(
        suite_name.c_str(),
        test_name.c_str(),
        nullptr,
        test_name.c_str(),
        __FILE__,
        __LINE__,
        // Important to use the fixture type as the return type here.
        [=]() -> Conv2dTestFixture<TConv2DTester, true, Threaded>* {
          return new Conv2dShortExecuteTest<TConv2DTester, Threaded>(
              BatchCount,
              GroupCount,
              InputChannels,
              InputHeight,
              InputWidth,
              FilterCount,
              KernelHeight,
              KernelWidth,
              PaddingLeftHeight,
              PaddingLeftWidth,
              PaddingRightHeight,
              PaddingRightWidth,
              DilationHeight,
              DilationWidth,
              StrideHeight,
              StrideWidth);
        });
    return 1;
  }

  static size_t RegisterShortExecTests() {
    size_t test_registered = 0;
    for (unsigned i = 1; i < 256; i <<= 1) {
      RegisterSingleTest(1, 1, 16, i, i, 32, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1);
      RegisterSingleTest(1, 1, 16, i, i, 32, 3, 3, 0, 0, 0, 0, 1, 1, 2, 2);
      RegisterSingleTest(1, 1, 16, i, i, 32, 3, 3, 0, 0, 0, 0, 2, 2, 1, 1);
      RegisterSingleTest(1, 1, 16, i, i, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1);
      RegisterSingleTest(1, 1, 16, i, i, 32, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1);
      RegisterSingleTest(1, 1, 16, i, i, 32, i, 1, 0, 0, 0, 0, 1, 1, 1, 1);
      RegisterSingleTest(1, 1, 16, i, i, 32, 1, i, 0, 0, 0, 0, 1, 1, 1, 1);
    }
    return test_registered;
  }

 private:
  size_t BatchCount_;
  size_t GroupCount_;
  size_t InputChannels_;
  size_t InputHeight_;
  size_t InputWidth_;
  size_t FilterCount_;
  size_t KernelHeight_;
  size_t KernelWidth_;
  size_t PaddingLeftHeight_;
  size_t PaddingLeftWidth_;
  size_t PaddingRightHeight_;
  size_t PaddingRightWidth_;
  size_t DilationHeight_;
  size_t DilationWidth_;
  size_t StrideHeight_;
  size_t StrideWidth_;
};

template <typename TConv2DTester, bool Threaded>
class Conv2dLongExecuteTest : public Conv2dTestFixture<TConv2DTester, false, Threaded> {
 public:
  void TestBody() override {
    conv2d_tester->ExecuteLong();
  }

  static size_t RegisterLongExecute() {
    static const std::string suite_name =
        std::string(TConv2DTester::GetSuitePrefix()) + (Threaded ? "_Long_Threaded" : "_Long_SingleThread");

    testing::RegisterTest(
        suite_name.c_str(),
        "LongExecute",
        nullptr,
        "LongExecute",
        __FILE__,
        __LINE__,
        // Important to use the fixture type as the return type here.
        [=]() -> Conv2dTestFixture<TConv2DTester, false, Threaded>* {
          return new Conv2dLongExecuteTest<TConv2DTester, Threaded>();
        });
    return 1;
  }
};
