CC ?= cc
CFLAGS ?= -O3 -Wall -Wextra -pedantic -std=c99 -Iinclude
LDFLAGS ?= -lm

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LIB_EXT := dylib
    SHARED_FLAGS := -dynamiclib -fPIC -install_name @rpath/libreflex.dylib
else
    LIB_EXT := so
    SHARED_FLAGS := -shared -fPIC
endif

BUILD_DIR := build
LIB_TARGET := $(BUILD_DIR)/libreflex.$(LIB_EXT)
BENCH_TARGET := $(BUILD_DIR)/reflex_bench

.PHONY: all clean test

all: $(LIB_TARGET) $(BENCH_TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(LIB_TARGET): src/reflex.c src/reflex_simd.c src/reflex_hnsw.c src/reflex_pq.c include/reflex.h include/reflex_simd.h include/reflex_hnsw.h include/reflex_pq.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SHARED_FLAGS) src/reflex.c src/reflex_simd.c src/reflex_hnsw.c src/reflex_pq.c -o $(LIB_TARGET) $(LDFLAGS)

$(BENCH_TARGET): src/main.c src/reflex.c src/reflex_simd.c src/reflex_hnsw.c src/reflex_pq.c include/reflex.h include/reflex_simd.h include/reflex_hnsw.h include/reflex_pq.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) src/main.c src/reflex.c src/reflex_simd.c src/reflex_hnsw.c src/reflex_pq.c -o $(BENCH_TARGET) $(LDFLAGS)

test: $(BENCH_TARGET)
	./$(BENCH_TARGET)

clean:
	rm -rf $(BUILD_DIR)
