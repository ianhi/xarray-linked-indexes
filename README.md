# Xarray Linked Indexes

Custom Indexes to support linked indexers for dimension and non-dimension coords in Xarray.

## Contributing

Contributions are welcome! Docs, code or otherwise. Feel free to ask for help or advice in issues if you get stuck.

### Development Setup

```bash
uv sync --group dev
uv run pytest
```

For jupyterlab-vim:

```bash
uv sync --group dev --group vim
```

### Building the Docs

The documentation is built using [MyST](https://mystmd.org/). To build the docs locally:

```bash
# Install mystmd globally (or use npx)
npm install -g mystmd

# Build and serve docs locally
myst start
```

This will start a local server at http://localhost:3000 with live reloading.

To build static HTML:

```bash
myst build --html
```

The output will be in `docs/_build/html/`.
