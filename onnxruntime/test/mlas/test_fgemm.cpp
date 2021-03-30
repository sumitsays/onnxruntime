#include "test_util.h"
#include "gtest/gtest.h"
#include <memory>
#include <sstream>

template <typename T, bool Packed>
class FGemmTestedContext {
 public:
  void TestGemm(CBLAS_TRANSPOSE TransA,
                CBLAS_TRANSPOSE TransB,
                size_t M,
                size_t N,
                size_t K,
                float alpha,
                const T* A,
                size_t lda,
                const T* B,
                size_t ldb,
                float beta,
                T* C,
                size_t ldc,
                MLAS_THREADPOOL* threadpool) {
    if (Packed) {
      size_t PackedBSize = MlasGemmPackBSize(N, K);
      void* PackedB = BufferBPacked.GetBuffer(PackedBSize, true);
      MlasGemmPackB(TransB, N, K, B, ldb, PackedB);
      MlasGemm(TransA, M, N, K, T(alpha), A, lda, PackedB, T(beta), C, ldc, threadpool);
    } else {
      MlasGemm(TransA, TransB, M, N, K, T(alpha), A, lda, B, ldb, T(beta), C, ldc, threadpool);
    }
  }

 private:
  // No hurt when not used in non pack gemm.
  MatrixGuardBuffer<uint8_t> BufferBPacked;
};

template <typename T, bool Packed>
class MlasFgemmTest {
 public:
  void Test(size_t M, size_t N, size_t K, float alpha, float beta, MLAS_THREADPOOL* threadpool) {
    Test(false, false, M, N, K, alpha, beta, threadpool);
    Test(false, true, M, N, K, alpha, beta, threadpool);
    Test(true, false, M, N, K, alpha, beta, threadpool);
    Test(true, true, M, N, K, alpha, beta, threadpool);
  }

  void Test(bool trans_a, bool trans_b, size_t M, size_t N, size_t K, float alpha, float beta, MLAS_THREADPOOL* threadpool) {
    //
    // Skip the test if the B buffer cannot be packed.
    //

    if (Packed && (N == 0 || K == 0)) {
      return;
    }

    const T* A = BufferA.GetBuffer(K * M);
    const T* B = BufferB.GetBuffer(N * K);
    T* C = BufferC.GetBuffer(N * M);
    T* CReference = BufferCReference.GetBuffer(N * M);

    Test(trans_a ? CblasTrans : CblasNoTrans,
         trans_b ? CblasTrans : CblasNoTrans,
         M, N, K, alpha, A, trans_a ? M : K, B, trans_b ? K : N,
         beta, C, CReference, N, threadpool);
  }

  void Test(CBLAS_TRANSPOSE TransA,
            CBLAS_TRANSPOSE TransB,
            size_t M,
            size_t N,
            size_t K,
            float alpha,
            const T* A,
            size_t lda,
            const T* B,
            size_t ldb,
            float beta,
            T* C,
            T* CReference,
            size_t ldc,
            MLAS_THREADPOOL* threadpool) {
    std::fill_n(C, M * N, -0.5f);
    std::fill_n(CReference, M * N, -0.5f);

    PackedContext.TestGemm(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, threadpool);
    ReferenceGemm(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, CReference, ldc);

    for (size_t m = 0, f = 0; m < M; m++) {
      for (size_t n = 0; n < N; n++) {
        // Sensitive to comparing positive/negative zero.
        if (C[f] != CReference[f]) {
          ASSERT_EQ(C[f], CReference[f])
              << " Diff @[" << m << ", " << n << "] f=" << f << ", "
              << (Packed ? "Packed" : "NoPack") << "."
              << (threadpool == nullptr ? "NoThread" : "Threaded") << "/"
              << (TransA == CblasTrans ? "TransA" : "A") << "/"
              << (TransB == CblasTrans ? "TransB" : "B") << "/"
              << "M:" << M << "xN:" << N << "xK:" << K << "/"
              << "Alpha:" << alpha << "/"
              << "Beta:" << beta;
        }
      }
    }
  }

  void
  ReferenceGemm(
      CBLAS_TRANSPOSE TransA,
      CBLAS_TRANSPOSE TransB,
      size_t M,
      size_t N,
      size_t K,
      float alpha,
      const T* A,
      size_t lda,
      const T* B,
      size_t ldb,
      float beta,
      T* C,
      size_t ldc) {
    if (TransA == CblasNoTrans) {
      if (TransB == CblasNoTrans) {
        for (size_t m = 0; m < M; m++) {
          for (size_t n = 0; n < N; n++) {
            const T* a = A + (m * lda);
            const T* b = B + n;
            T* c = C + (m * ldc) + n;
            T sum = 0.0f;

            for (size_t k = 0; k < K; k++) {
              sum += (*b * *a);
              b += ldb;
              a += 1;
            }

            *c = (*c * beta) + (sum * alpha);
          }
        }

      } else {
        for (size_t m = 0; m < M; m++) {
          for (size_t n = 0; n < N; n++) {
            const T* a = A + (m * lda);
            const T* b = B + (n * ldb);
            T* c = C + (m * ldc) + n;
            T sum = 0.0f;

            for (size_t k = 0; k < K; k++) {
              sum += (*b * *a);
              b += 1;
              a += 1;
            }

            *c = (*c * beta) + (sum * alpha);
          }
        }
      }

    } else {
      if (TransB == CblasNoTrans) {
        for (size_t m = 0; m < M; m++) {
          for (size_t n = 0; n < N; n++) {
            const T* a = A + m;
            const T* b = B + n;
            T* c = C + (m * ldc) + n;
            T sum = 0.0f;

            for (size_t k = 0; k < K; k++) {
              sum += (*b * *a);
              b += ldb;
              a += lda;
            }

            *c = (*c * beta) + (sum * alpha);
          }
        }

      } else {
        for (size_t m = 0; m < M; m++) {
          for (size_t n = 0; n < N; n++) {
            const T* a = A + m;
            const T* b = B + (n * ldb);
            T* c = C + (m * ldc) + n;
            T sum = 0.0f;

            for (size_t k = 0; k < K; k++) {
              sum += (*b * *a);
              b += 1;
              a += lda;
            }

            *c = (*c * beta) + (sum * alpha);
          }
        }
      }
    }
  }

  MatrixGuardBuffer<T> BufferA;
  MatrixGuardBuffer<T> BufferB;
  MatrixGuardBuffer<T> BufferC;
  MatrixGuardBuffer<T> BufferCReference;
  FGemmTestedContext<T, Packed> PackedContext;

  void ExecuteLong(MLAS_THREADPOOL* threadpool) {
    static const float multipliers[] = {0.0f, -0.0f, 0.25f, -0.5f, 1.0f, -1.0f};

    for (size_t N = 1; N < 128; N++) {
      for (size_t K = 1; K < 128; K++) {
        for (size_t a = 0; a < _countof(multipliers); a++) {
          for (size_t b = 0; b < _countof(multipliers); b++) {
            Test(1, N, K, multipliers[a], multipliers[b], threadpool);
            Test(N, 1, K, multipliers[a], multipliers[b], threadpool);
          }
        }
      }
    }

    for (size_t a = 0; a < _countof(multipliers); a++) {
      float alpha = multipliers[a];

      for (size_t b = 0; b < _countof(multipliers); b++) {
        float beta = multipliers[b];

        for (size_t M = 16; M < 160; M += 32) {
          for (size_t N = 16; N < 160; N += 32) {
            static const size_t ks[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 32, 48, 64, 118, 119, 120, 121, 122, 160, 240, 320};
            for (size_t k = 0; k < _countof(ks); k++) {
              size_t K = ks[k];

              Test(M, N, K, alpha, beta, threadpool);
              Test(M + 1, N, K, alpha, beta, threadpool);
              Test(M, N + 1, K, alpha, beta, threadpool);
              Test(M + 1, N + 1, K, alpha, beta, threadpool);
              Test(M + 3, N + 2, K, alpha, beta, threadpool);
              Test(M + 4, N, K, alpha, beta, threadpool);
              Test(M, N + 4, K, alpha, beta, threadpool);
              Test(M + 4, N + 4, K, alpha, beta, threadpool);
              Test(M + 3, N + 7, K, alpha, beta, threadpool);
              Test(M + 8, N, K, alpha, beta, threadpool);
              Test(M, N + 8, K, alpha, beta, threadpool);
              Test(M + 12, N + 12, K, alpha, beta, threadpool);
              Test(M + 13, N, K, alpha, beta, threadpool);
              Test(M, N + 15, K, alpha, beta, threadpool);
              Test(M + 15, N + 15, K, alpha, beta, threadpool);
            }
          }
          printf("a %zd/%zd b %zd/%zd M %zd\n", a, _countof(multipliers), b, _countof(multipliers), M);
        }
      }
    }

    for (size_t M = 0; M < 160; M++) {
      for (size_t N = 0; N < 160; N++) {
        for (size_t K = 0; K < 160; K++) {
          Test(M, N, K, 1.0f, 0.0f, threadpool);
        }
      }
      printf("M %zd\n", M);
    }

    for (size_t M = 160; M < 320; M += 24) {
      for (size_t N = 112; N < 320; N += 24) {
        for (size_t K = 0; K < 16; K++) {
          Test(M, N, K, 1.0f, 0.0f, threadpool);
        }
        for (size_t K = 16; K < 160; K += 32) {
          Test(M, N, K, 1.0f, 0.0f, threadpool);
        }
      }
      printf("M %zd\n", M);
    }
  }
};

template <typename T, bool Packed>
class FGemmTestFixture : public testing::Test {
 public:
  static void SetUpTestSuite() {
    mlas_gemm_tester = new MlasFgemmTest<T, Packed>();
  };

  static void TearDownTestSuite() {
    delete mlas_gemm_tester;
    mlas_gemm_tester = nullptr;
  };

  static MlasFgemmTest<T, Packed>* mlas_gemm_tester;
};

MlasFgemmTest<float, false>* FGemmTestFixture<float, false>::mlas_gemm_tester(nullptr);
MlasFgemmTest<float, true>* FGemmTestFixture<float, true>::mlas_gemm_tester(nullptr);

// Short Execute treat each test seperately by all parameters.
template <bool Packed>
class FgemmShortExecuteTests : public FGemmTestFixture<float, Packed> {
 public:
  explicit FgemmShortExecuteTests(
      bool trans_a, bool trans_b, size_t M, size_t N, size_t K,
      float alpha, float beta, MLAS_THREADPOOL* threadpool)
      : trans_a_(trans_a), trans_b_(trans_b), M_(M), N_(N), K_(K), alpha_(alpha), beta_(beta), threadpool_(threadpool) {
  }

  void TestBody() override {
    mlas_gemm_tester->Test(trans_a_, trans_b_, M_, N_, K_, alpha_, beta_, threadpool_);
  }

  static size_t RegisterTest(
      bool trans_a, bool trans_b, size_t M, size_t N, size_t K,
      float alpha, float beta, MLAS_THREADPOOL* threadpool) {
    std::stringstream ss;
    ss << (trans_a ? "TransA" : "A") << "/"
       << (trans_b ? "TransB" : "B") << "/"
       << "M:" << M << "xN:" << N << "xK:" << K << "/"
       << "Alpha:" << alpha << "/"
       << "Beta:" << beta;
    auto name = ss.str();
    std::string prefix = (threadpool == nullptr) ? "SingleThread/" : "Threaded/";

    testing::RegisterTest(
        Packed ? "FgemmPack_ShortExec" : "FgemmNoPack_ShortExec",
        (prefix + name).c_str(),
        nullptr,
        name.c_str(),
        __FILE__,
        __LINE__,
        // Important to use the fixture type as the return type here.
        [=]() -> FGemmTestFixture<float, Packed>* {
          return new FgemmShortExecuteTests<Packed>(
              trans_a, trans_b, M, N, K, alpha, beta, threadpool);
        });
    return 1;
  }

  static size_t RegisterTest(size_t M, size_t N, size_t K, float alpha, float beta) {
    auto count = RegisterTest(false, false, M, N, K, alpha, beta, nullptr);
    count += RegisterTest(false, true, M, N, K, alpha, beta, nullptr);
    count += RegisterTest(true, false, M, N, K, alpha, beta, nullptr);
    count += RegisterTest(true, true, M, N, K, alpha, beta, nullptr);
    const auto tp = GetMlasThreadPool();
    if (tp != nullptr) {
      count = RegisterTest(false, false, M, N, K, alpha, beta, tp);
      count += RegisterTest(false, true, M, N, K, alpha, beta, tp);
      count += RegisterTest(true, false, M, N, K, alpha, beta, tp);
      count += RegisterTest(true, true, M, N, K, alpha, beta, tp);
    }
    return count;
  }

  static size_t Register() {
    size_t test_registered = 0;
    for (size_t b = 0; b < 16; b++) {
      test_registered += FgemmShortExecuteTests<Packed>::RegisterTest(b, b, b, 1.0f, 0.0f);
    }
    for (size_t b = 16; b <= 256; b <<= 1) {
      test_registered += FgemmShortExecuteTests<Packed>::RegisterTest(b, b, b, 1.0f, 0.0f);
      test_registered++;
    }
    for (size_t b = 256; b < 320; b += 32) {
      test_registered += FgemmShortExecuteTests<Packed>::RegisterTest(b, b, b, 1.0f, 0.0f);
      test_registered++;
    }

    test_registered += FgemmShortExecuteTests<Packed>::RegisterTest(128, 3072, 768, 1.0f, 0.0f);
    test_registered += FgemmShortExecuteTests<Packed>::RegisterTest(128, 768, 3072, 1.0f, 0.0f);
    return test_registered;
  }

 private:
  bool trans_a_, trans_b_;
  size_t M_, N_, K_;
  float alpha_, beta_;
  MLAS_THREADPOOL* threadpool_;
};

static bool s_gestisted_short =
    LongShortExecuteManager::instance().AddShortExcuteTests(FgemmShortExecuteTests<false>::Register) &&
    LongShortExecuteManager::instance().AddShortExcuteTests(FgemmShortExecuteTests<true>::Register);

// Long Execute test. It is too heavy register each single test, treat long execute big groups.
template <bool Packed>
class FgemmLongExecuteTests : public FGemmTestFixture<float, Packed> {
 public:
  explicit FgemmLongExecuteTests(MLAS_THREADPOOL* threadpool) : threadpool_(threadpool) {
  }

  void TestBody() override {
    mlas_gemm_tester->ExecuteLong(threadpool_);
  }

  static size_t RegisterTest(MLAS_THREADPOOL* threadpool) {
    std::string prefix = (threadpool == nullptr) ? "SingleThread" : "Threaded";

    testing::RegisterTest(
        Packed ? "FgemmPack_LongExec" : "FgemmNoPack_LongExec",
        prefix.c_str(),
        nullptr,
        prefix.c_str(),
        __FILE__,
        __LINE__,
        // Important to use the fixture type as the return type here.
        [=]() -> FGemmTestFixture<float, Packed>* {
          return new FgemmLongExecuteTests<Packed>(threadpool);
        });
    return 1;
  }

  static size_t Register(void) {
    auto count = RegisterTest(nullptr);
    if (GetMlasThreadPool() != nullptr) {
      count += RegisterTest(GetMlasThreadPool());
    }
    return count;
  }

 private:
  MLAS_THREADPOOL* threadpool_;
};

static bool s_gestisted_long =
    LongShortExecuteManager::instance().AddLongExcuteTests(FgemmLongExecuteTests<false>::Register) &&
    LongShortExecuteManager::instance().AddLongExcuteTests(FgemmLongExecuteTests<true>::Register);
