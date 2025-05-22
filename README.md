# gil.md

### Make sure you have a python3.13 without GIL build

bash ./configure --disable-gil

#### run it

```bash
# With GIL enabled
PYTHON_GIL=1 python david_gilmour.py
```
```bash
# Without gil
PYTHON_GIL=0 python david_gilmour.py
```
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ehzawad/gil.md)
