# Python 3.13.2 NoGIL with OpenSSL 3.x

This repository provides instructions for building Python 3.13.2 with the `--disable-gil` option on macOS when using OpenSSL 3.x. This solves the compatibility issues that prevent Python 3.13.2 from building with newer OpenSSL versions.

## The Problem

Python 3.13.2 expects OpenSSL 1.1.1 symbols that don't exist in OpenSSL 3.x. When attempting to build Python 3.13.2 with OpenSSL 3.x, you'll encounter errors like:

```
[ERROR] *hashlib failed to import: dlopen(.../_hashlib.cpython-313t-darwin.so, 0x0002): symbol not found in flat namespace '_EVP_MD_CTX_get0_md'
[ERROR] *ssl failed to import: dlopen(.../_ssl.cpython-313t-darwin.so, 0x0002): symbol not found in flat namespace '_SSL_SESSION_get_time_ex'
Could not build the ssl module!
Python requires a OpenSSL 1.1.1 or newer
```

## The Solution

A simple patch to the Python source code fixes the compatibility issues with OpenSSL 3.x.

## Prerequisites

- macOS (tested on macOS 15.4)
- Xcode Command Line Tools (`xcode-select --install`)
- Homebrew installed packages:
  - `brew install openssl@3`
  - `brew install gdbm`
  - `brew install sqlite`
  - `brew install readline`
  - `brew install xz`  # for lzma support
- Python 3.13.2 source code

## Installation Steps

1. Download and extract the Python 3.13.2 source code:

```bash
cd ~/Downloads
curl -O https://www.python.org/ftp/python/3.13.2/Python-3.13.2.tar.xz
tar -xf Python-3.13.2.tar.xz
cd Python-3.13.2
```

2. Create and apply the OpenSSL 3.x compatibility patch:

```bash
# Create the patch file (run this from the Python source directory)
cat > openssl3_compat.patch << 'EOF'
diff --git a/Modules/_ssl.c b/Modules/_ssl.c
index 1a5f2d8098..a81de2f4e9 100644
--- a/Modules/_ssl.c
+++ b/Modules/_ssl.c
@@ -108,7 +108,7 @@
 
 /* OpenSSL 1.1.0+ */
 #if (OPENSSL_VERSION_NUMBER >= 0x10100000L) && !defined(LIBRESSL_VERSION_NUMBER)
-#define OPENSSL_VERSION_1_1 1
+#define OPENSSL_VERSION_1_1_PLUS 1
 #endif
 
 /* OpenSSL 1.1.1+ */
EOF

# Apply the patch
patch -p1 < openssl3_compat.patch
```

3. Configure and build Python with explicit package paths and no-GIL option:

```bash
# Configure with all necessary paths and flags
CFLAGS="-I$(brew --prefix openssl@3)/include" \
LDFLAGS="-L$(brew --prefix openssl@3)/lib" \
GDBM_CFLAGS="-I$(brew --prefix gdbm)/include" \
GDBM_LIBS="-L$(brew --prefix gdbm)/lib -lgdbm" \
./configure --with-openssl="$(brew --prefix openssl@3)" \
            --with-system-libmpdec \
            --disable-gil \
            --prefix=/Users/YOUR_USERNAME/nogil

# For debugging builds, add --with-pydebug to the configure options

# Build and install
make -j$(sysctl -n hw.ncpu)
make install
```

4. Add the new Python to your PATH:

```bash
# Add this line to your ~/.zshrc or ~/.bash_profile
export PATH="/Users/YOUR_USERNAME/nogil/bin:$PATH"

# Source your profile or restart your terminal
source ~/.zshrc  # or ~/.bash_profile
```

5. Verify the installation:

```bash
# Check if Python works
python3 -V

# Verify OpenSSL module works
python3 -c "import ssl; print(ssl.OPENSSL_VERSION)"

# Verify that GIL is disabled
python3 -c "import sys; print('GIL is disabled' if not sys.flags.gil_enabled else 'GIL is enabled')"
```

## Explanation of the Patch

In simple terms, here's what the patch does:

When Python 3.13.2 tries to build with OpenSSL, it checks what version of OpenSSL you have. If it finds OpenSSL 1.1.0 or newer, it sets a flag called `OPENSSL_VERSION_1_1`. 

The problem is that elsewhere in Python's code, when this flag is set, it tries to use specific OpenSSL functions that existed in OpenSSL 1.1.x but were renamed or changed in OpenSSL 3.x (like the `_EVP_MD_CTX_get0_md` and `_SSL_SESSION_get_time_ex` functions that were failing).

By changing the flag name from `OPENSSL_VERSION_1_1` to `OPENSSL_VERSION_1_1_PLUS`, we trick Python's code - when it checks for the flag `OPENSSL_VERSION_1_1`, it won't find it, so it falls back to using older, more compatible OpenSSL function calls that still work in OpenSSL 3.x.

It's basically telling Python "don't use those newer OpenSSL 1.1 features that changed in OpenSSL 3.x - stick with the more compatible approach."

Technically, the patch modifies a macro definition in the `_ssl.c` file:

1. It changes `OPENSSL_VERSION_1_1` to `OPENSSL_VERSION_1_1_PLUS`
2. This makes Python use the OpenSSL 3.x API in a way that's compatible with Python's expectations

## Troubleshooting

If you encounter any issues:

1. Make sure you're running the commands from the Python source directory
2. Ensure all required packages are installed:
   ```bash
   brew info openssl@3 gdbm sqlite readline xz
   ```
3. Try clearing any previous build artifacts: `make clean` before rebuilding
4. Check that the patch was applied successfully: `grep -A 2 -B 2 "OPENSSL_VERSION_1_1_PLUS" Modules/_ssl.c`
5. If you see errors about missing modules, check module-specific dependencies:
   ```bash
   # In the Python source directory
   cat Modules/Setup.dist  # This shows what optional modules can be built
   ```
6. For other build issues, consult the detailed build logs in the `config.log` file

## Notes

- This method is tested with Python 3.13.2 and OpenSSL 3.4.1
- The patched Python should work for most purposes, but thoroughly test your applications
- This is an alternative to using pyenv, which normally handles these compatibility issues automatically

## License

This README and the patch are provided under the MIT License. Python itself is under the PSF License.
