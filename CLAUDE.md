# SOUL

Propose is required for every critical action. Propose only what you will do but not implications, then literally present 'Say `sanction` to act'. 
Only act on this proposal if the user says exact word 'sanction' or 'sanctioned'. 
 
To act, only implement what the user sanctioned / said directly inside the prompt. 
If error happens in a critical action, the fix needs proposing + sanctioning again. Otherwise, fix in action directly. 

To inform, only answer what the user asked. 
Your first sentence must be a standalone answer in less than 10 words. Prefer literal phrase over mannered prose. 

Critical action: 
+ a command **known** to be longer than 5 seconds
+ apply `Write/Edit` tool to edit production source code, i.e. git tracked code (don't apply `Bash`)

# CODE

To write production code, present code skeleton listing for sanctioning; Only after user sanctioning, you code; Section header file/module paths, content is a bare code block of signatures + SHAME(...) tag when syntax rules match; Do not attach reasoning / prose / comments / logic description; If user asks, use a dedicated turn to reply
+ **CPP**
    + full item removal: `-` mark before name-only item (`-class Thing`), elide body; for partial update, apply following rules
    + `#define` / file scope `constexpr` / `using` : present full
    + `class` / `struct` / `enum` : full, `+` mark field/variant/method addition (`+Type field;`), `-` mark removal, method follows function convention
    + function: only declaration signature in full, `template` header included, `+` mark arg addition (`+Type arg`), `-` mark removal, elide body
    + out-of-line definition : inferrable from the declaration, omit dedicated presentation
    + `#include` / `namespace` : inferrable from item path (`ns::xx`), omit dedicated presentation
+ **CUDA**
    + inherit **CPP**, CUDA decoration is part of signature

To write a syntax item in production code, use the following convention:
+ **CPP**
    + file name : one word, or two word `flatcase.{cc,cxx,cpp,hh,hpp,hxx,...}`
    + `#define` / file scope `constexpr` / block scope compile time `constexpr` name : one word or two word `SNAKE_CAPITAL_CASE`, a macro carries its component as prefix
    + `class` / `struct` / `enum` name : normal `CamelCase`, a hardware prefix stays an acronym
        + field name : one word, or two word `flatcase`, same `struct` / `class` field names should have all same length
        + `enum` variant name : one word, or two word `SNAKE_CASE`, per enum variants all same length
    + no nested function body inside `class`
    + method / free function / local name : one word, or two word `flatcase`; a file scope free function is `static`
    + `template` parameter name : one word `SNAKE_CAPITAL_CASE`, a `bool` predicate parameter one word `flatcase`
    + `comment` : always use `/** @brief */` on the declaration, one line each for `@tparam` / `@param` / `@return`; in function bodies only numbered step markers `// [1]`; per block at most 60 words
    + `comment` : add literal tags in comments to functions more than 60 lines `SHAME(TALLFUNC)` / 100 chars `SHAME(WIDEFUNC)` / 6 args `SHAME(MANYARG)`
+ **CUDA** (inherit **CPP**)
    + `comment` : a kernel documents its index layout, not its arithmetic; the buffer order it maintains belongs in the `@class` block

To write a correctness test for production code, follow listed conventions; test logic should be simpler than code; test is code, so it requires the same present + sanction process:
+ **CPP**
    + place `unittest/test_<...>.{cc,cpp,cxx}`, `main` function runs test suite, return error code (0 => success)
    + register in `CMakeLists.txt` by `add_executable` + `add_test`
+ **CUDA**
    + inherit **CPP**
    + sweep whole launch shape space, assert on host after `cudaDeviceSynchronize` and copy back, never inside a kernel

# GIT

Commit message: `feature/refactor/chore/test/fix`, top-level crate/module/file path in parentheses, e.g. `feature(some_crate): ...`, only one line
Just push: code only goes into git push, never copy code files directly unless you are sure that the script can only ever exists on the server and never enter github
Commit as you go: end each turn with a commit whenever there is code change
Be plain: `git ...` instead of `git ... && something else && ... git ...`; otherwise it gets rejected
No worktree: do not create worktrees, user usually don't ask for overlapping changes; if there are, communicate with other agent sessions
