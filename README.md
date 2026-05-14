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
thread_pool ml_ddos threads=4 max_queue=65536;

http {
    ml_ddos_path /etc/nginx/model.onnx;
    server {
        listen 80;
        server_name localhost;

        location /ml_async {
            ml_ddos on thread=ml_ddos block=0.8 limit=0.6;
            proxy_pass http://backend_upstream;
        }

        location /ml_async_sampling {
            ml_ddos on thread=ml_ddos mode=sampling;
            default_type text/plain;
            return 200 "ML\n";
        }
        
        location /ml_sync {
            # Default options: without thread pool, mode=strict, block=0.85, limit=0.65
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
The following benchmarks were conducted using [wrk](https://github.com/wg/wrk) with the parameters `-t16 -c1000 -d10s`on a local development environment.

Single Worker Performance (`worker_processes 1;`)

|              |RPS     |Avg Latency|Latency Stdev|
|--------------|--------|-----------|-------------|
|Disabled      |67627.81|34.66 ms   |150.14 ms    |
|Async         |56103.75|43.71 ms   |157.97 ms    |
|Async sampling|35100.64|40.53 ms   |136.60 ms    |
|Sync (strict) |30475.06|33.22 ms   |87.10 ms     |

Multi-Worker Performance (`worker_processes auto;`)

|              |RPS      |Avg Latency|Latency Stdev|
|--------------|---------|-----------|-------------|
|Disabled      |322997.38|3.20 ms    |2.31 ms      |
|Async         |109816.13|9.03 ms    |3.42 ms      |
|Async sampling|182356.31|5.62 ms    |3.65 ms      |
|Sync (strict) |176545.12|5.64 ms    |2.55 ms      |
