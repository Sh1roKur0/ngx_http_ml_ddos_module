# ML DDOS Module for Nginx

This module provides advanced DDoS protection by constructing time-series data and identifying traffic patterns using an ONNX-based model (Inference). By analyzing request sequences, it can distinguish between legitimate users and automated attack patterns that traditional rate-limiters might miss.

### Build
To build the module, ensure you have meson and ninja installed, along with the ONNX Runtime C API.

```sh
# Setup the build directory
meson setup build

# Compile the module
# Use --verbose if you need to debug the compilation process
meson compile -C build

# Install the module as modules/ngx_http_ddos_module.so to prefix
# /etc/nginx is the default prefix, use --prefix=/path/to/dir at setup to change
sudo meson install -C build
```

##### Useful Build Flags
You can pass these flags to `meson setup` to customize the build environment (full list [here](https://mesonbuild.com/Builtin-options.html)):

| Flag | Values | Description |
| --- | --- |--- |
| --buildtype | debug, release, plain | Overall build setup (e.g., debug includes symbols). |
| -Doptimization | 0, 1, 2, 3, s, g | Code optimization level (s focuses on binary size). |
| -Dprefix | /path/to/dir | Installation prefix for the compiled binaries. |
| -Dwerror | true, false | If true, stops the build on any compiler warning. |

### Usage
1. Load the Module: First, load the module at the top of your `nginx.conf`:

``` nginx
load_module modules/ngx_http_ml_ddos_module.so;
```
2. Configure the Directive: Enable the protection within a server or location block. You can optionally specify the path to your ONNX model. If no path is provided, it defaults to /etc/nginx/model.onnx
``` nginx
server {
    listen 80;
    server_name localhost;

    location / {
        # Use the default model path
        ngx_http_ml_ddos;

        # OR specify a custom model path
        # ngx_http_ml_ddos /etc/nginx/models/custom_ddos_v1.onnx;

        proxy_pass http://backend_upstream;
    }
}
```
