message(STATUS "Build with contrib.kmeans")
file(GLOB KMEANS_CONTRIB_SRC src/runtime/contrib/kmeans/*.cc)
list(APPEND RUNTIME_SRCS ${KMEANS_CONTRIB_SRC})
