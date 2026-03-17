# C++ Style Rules

- Use `#pragma once` (not `#ifndef` guards)
- Prefer `const`/`constexpr` everywhere it fits; mark all read-only arguments `const&`
- Use `[[nodiscard]]` on functions returning error codes or computed values
- Prefer `std::size_t` for indices/sizes; avoid signed/unsigned comparison warnings
- Use `auto` only when the type is obvious from the right-hand side (e.g. iterators, `.begin()`)
- Naming: `snake_case` for variables/functions, `PascalCase` for types/classes, `ALL_CAPS` for constants
- Every new header goes in `include/`; every demo/test goes in `examples/`
- Never add external dependencies without discussion — this project is header-only (ONNX Runtime is the planned exception)
- Keep each header focused on one abstraction; avoid god headers
- Avoid raw `new`/`delete`; prefer `std::vector`, `std::unique_ptr`, stack allocation
- All RNG must use `std::mt19937` seeded from `std::random_device`
