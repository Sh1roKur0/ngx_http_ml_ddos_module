# ML DDOS Module for Nginx

This module provides advanced DDoS protection by constructing time-series data and identifying traffic patterns using an ONNX-based model (Inference). By analyzing request sequences, it can distinguish between legitimate users and automated attack patterns that traditional rate-limiters might miss.

### Build
To build the module, use the default method of building with sources (or dynamic). It uses `pkg-config` to find the `onnxruntime` library.

```sh
# Setup the build directory
$ ./configure --with-dynamic-module=<PATH_TO_MODULE> --with-compat

# Compile the module
$ make modules
```

### Usage
1. Load the Module: First, load the module at the top of your `nginx.conf`:

``` nginx
load_module modules/ngx_http_ml_ddos_module.so;
```

2. Configure the server:
``` nginx
# Must use the name ml_ddos
thread_pool ml_ddos threads=16 max_queue=65536;

http {
    ml_ddos_path /etc/nginx/model.onnx;
    server {
        listen 80;
        server_name localhost;

        location / {
            ml_ddos on;
            proxy_pass http://backend_upstream;
        }

        location /off {
            # You can disable the ml_ddos explicitly, as it is disabled by default
            ml_ddos off;
            proxy_pass http://frontend_upstream;
        }
    }
}
```

### Benchmark
Using [wrk](https://github.com/wg/wrk) with the options `-t16 -c1000 -d10s`, we obtained the following results:

|        |RPS        |Avg Latency|Latency Stdev|
|--------|-----------|-----------|-------------|
|Enabled |25363.97   |62.74 ms   |150.68 ms    |
|Disabled|68731.38   |33.41 ms   |132.33 ms    |
