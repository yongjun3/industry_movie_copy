#### Movie Recommendation Application

Using `docker` to build the application:
```
$ docker compose up --build
```

Call the recommendation:
```
$ curl http://127.0.0.1:8082/recommend/userid
```
