// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdio.h>
#include <memory.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <random>
#include <mlas.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#if !defined(MLAS_NO_ONNXRUNTIME_THREADPOOL)
#include "core/platform/threadpool.h"
#endif

#include "core/common/make_unique.h"

#if !defined(_countof)
#define _countof(_Array) (sizeof(_Array) / sizeof(_Array[0]))
#endif

MLAS_THREADPOOL* GetMlasThreadPool(void);

template <typename T>
class MatrixGuardBuffer {
 public:
  MatrixGuardBuffer() {
    _BaseBuffer = nullptr;
    _BaseBufferSize = 0;
    _ElementsAllocated = 0;
  }

  ~MatrixGuardBuffer(void) {
    ReleaseBuffer();
  }

  T* GetBuffer(size_t Elements, bool ZeroFill = false) {
    //
    // Check if the internal buffer needs to be reallocated.
    //

    if (Elements > _ElementsAllocated) {
      ReleaseBuffer();

      //
      // Reserve a virtual address range for the allocation plus an unmapped
      // guard region.
      //

      constexpr size_t BufferAlignment = 64 * 1024;
      constexpr size_t GuardPadding = 256 * 1024;

      size_t BytesToAllocate = ((Elements * sizeof(T)) + BufferAlignment - 1) & ~(BufferAlignment - 1);

      _BaseBufferSize = BytesToAllocate + GuardPadding;

#if defined(_WIN32)
      _BaseBuffer = VirtualAlloc(NULL, _BaseBufferSize, MEM_RESERVE, PAGE_NOACCESS);
#else
      _BaseBuffer = mmap(0, _BaseBufferSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif

      if (_BaseBuffer == nullptr) {
        ORT_THROW_EX(std::bad_alloc);
      }

      //
      // Commit the number of bytes for the allocation leaving the upper
      // guard region as unmapped.
      //

#if defined(_WIN32)
      if (VirtualAlloc(_BaseBuffer, BytesToAllocate, MEM_COMMIT, PAGE_READWRITE) == nullptr) {
        ORT_THROW_EX(std::bad_alloc);
      }
#else
      if (mprotect(_BaseBuffer, BytesToAllocate, PROT_READ | PROT_WRITE) != 0) {
        ORT_THROW_EX(std::bad_alloc);
      }
#endif

      _ElementsAllocated = BytesToAllocate / sizeof(T);
      _GuardAddress = (T*)((unsigned char*)_BaseBuffer + BytesToAllocate);
    }

    //
    //
    //

    T* GuardAddress = _GuardAddress;
    T* buffer = GuardAddress - Elements;

    if (ZeroFill) {
      std::fill_n(buffer, Elements, T(0));

    } else {
      const int MinimumFillValue = -23;
      const int MaximumFillValue = 23;

      int FillValue = MinimumFillValue;
      T* FillAddress = buffer;

      while (FillAddress < GuardAddress) {
        *FillAddress++ = (T)FillValue;

        FillValue++;

        if (FillValue > MaximumFillValue) {
          FillValue = MinimumFillValue;
        }
      }
    }

    return buffer;
  }

  void ReleaseBuffer(void) {
    if (_BaseBuffer != nullptr) {
#if defined(_WIN32)
      VirtualFree(_BaseBuffer, 0, MEM_RELEASE);
#else
      munmap(_BaseBuffer, _BaseBufferSize);
#endif

      _BaseBuffer = nullptr;
      _BaseBufferSize = 0;
    }

    _ElementsAllocated = 0;
  }

 private:
  size_t _ElementsAllocated;
  void* _BaseBuffer;
  size_t _BaseBufferSize;
  T* _GuardAddress;
};

typedef std::function<size_t()> TestRegistor;

// Singleton to let different part of test register.
class LongShortExecuteManager {
 public:
  static LongShortExecuteManager& instance(void);

  bool AddShortExcuteTests(TestRegistor test_registor) {
    short_execute_registers.push_back(test_registor);
    return true;
  }

  bool AddLongExcuteTests(TestRegistor test_registor) {
    long_execute_registers.push_back(test_registor);
    return true;
  }

  const std::list<TestRegistor>& GetRegistors(bool is_short_execute) const {
    return is_short_execute ? short_execute_registers : long_execute_registers;
  }

 private:
  LongShortExecuteManager(const LongShortExecuteManager&) = delete;
  LongShortExecuteManager& operator=(const LongShortExecuteManager&) = delete;

  LongShortExecuteManager() : short_execute_registers(), long_execute_registers() {
  }

  std::list<TestRegistor> short_execute_registers;
  std::list<TestRegistor> long_execute_registers;
};
