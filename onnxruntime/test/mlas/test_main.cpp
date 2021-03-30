#include "test_util.h"
#include "gtest/gtest.h"
#include <list>
#include <algorithm>

#if !defined(MLAS_NO_ONNXRUNTIME_THREADPOOL)

MLAS_THREADPOOL* GetMlasThreadPool(void) {
  static MLAS_THREADPOOL* threadpool = new onnxruntime::concurrency::ThreadPool(
      &onnxruntime::Env::Default(), onnxruntime::ThreadOptions(), nullptr, 2, true);
  return threadpool;
}

#else

MLAS_THREADPOOL* GetMlasThreadPool(void) {
  return nullptr;
}

#endif

LongShortExecuteManager& LongShortExecuteManager::instance() {
  static LongShortExecuteManager s_instance;
  return s_instance;
}

static size_t RegisterForShortOrLongExecute(bool is_short_execute) {
    const auto& test_registors = LongShortExecuteManager::instance().GetRegistors(is_short_execute);
    size_t test_count = 0;
    for (auto fn : test_registors) {
        test_count += fn();
    }
    return test_count;
}

int main(int argc, char** argv) {
  bool is_short_execute = (argc <= 1 || strcmp("--long", argv[1]) != 0);
  std::cout << "-------------------------------------------------------" << std::endl;
  if (is_short_execute) {
      std::cout << "----Running quick check mode. Enable more complete test" << std::endl;
      std::cout << "----  with '--long' as first argument!" << std::endl;
  }
  auto test_count = RegisterForShortOrLongExecute(is_short_execute);
  std::cout << "----Total " << test_count << " tests registered programmablely!" << std::endl;
  std::cout << "-------------------------------------------------------" << std::endl;

  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
