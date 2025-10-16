```bash
./configure \
  --prefix=/Users/ehz/opt/python/3.14.0 \
  --enable-optimizations \
  --with-lto
  CFLAGS="-march=native -mtune=native"
```

```bash
  make -j10
```

```bash
make alt install
```
