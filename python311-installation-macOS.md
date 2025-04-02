# 1. Download and extract Python 3.11.11
cd ~/Downloads
curl -O https://www.python.org/ftp/python/3.11.11/Python-3.11.11.tgz
tar -xf Python-3.11.11.tgz
cd Python-3.11.11

# 2. Apply the OpenSSL 3.x compatibility patch
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

# 3. Fix the mpdecimal configuration for macOS
if [ "$(uname -m)" = "arm64" ]; then
  # For Apple Silicon (arm64)
  sed -i '' 's/libmpdec_machine=universal/libmpdec_machine=uint128/g' configure
else
  # For Intel (x86_64)
  sed -i '' 's/libmpdec_machine=universal/libmpdec_machine=x64/g' configure
fi

# 4. Configure and build Python with optimizations
MPDECIMAL_PREFIX=$(brew --prefix mpdecimal)
CFLAGS="-I$(brew --prefix openssl@3)/include -I${MPDECIMAL_PREFIX}/include" \
LDFLAGS="-L$(brew --prefix openssl@3)/lib -L${MPDECIMAL_PREFIX}/lib" \
GDBM_CFLAGS="-I$(brew --prefix gdbm)/include" \
GDBM_LIBS="-L$(brew --prefix gdbm)/lib -lgdbm" \
./configure --with-openssl="$(brew --prefix openssl@3)" \
            --with-system-libmpdec="${MPDECIMAL_PREFIX}" \
            --enable-optimizations \
            --with-lto \
            --prefix=$HOME/.local

# 5. Build and install
make -j$(sysctl -n hw.ncpu)
make install
