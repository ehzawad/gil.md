# 1. Download and extract Python 3.11.11
cd ~/Downloads
curl -O https://www.python.org/ftp/python/3.11.11/Python-3.11.11.tgz
tar -xf Python-3.11.11.tgz
cd Python-3.11.11

# 2. Create and apply the OpenSSL 3.x compatibility patch
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

# 3. Configure and build Python with your custom prefix
CFLAGS="-I$(brew --prefix openssl@3)/include" \
LDFLAGS="-L$(brew --prefix openssl@3)/lib" \
GDBM_CFLAGS="-I$(brew --prefix gdbm)/include" \
GDBM_LIBS="-L$(brew --prefix gdbm)/lib -lgdbm" \
./configure --with-openssl="$(brew --prefix openssl@3)" \
            --with-system-libmpdec \
            --prefix=$HOME/.local

# 4. Build and install
make -j$(sysctl -n hw.ncpu)
make install
