export PORT=8080
docker build --debug -t mlzoomcamp2025_capstone2 -f Dockerfile .
docker run -p ${PORT}:${PORT} -it --rm mlzoomcamp2025_capstone2
