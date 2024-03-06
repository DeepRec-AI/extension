LIBNAME := gazer
OS ?= $(shell uname -s)


CXX ?= gcc
PYTHON ?= python

CFLAGS := -O3 -g \
	-DNDEBUG \
	-I$(LIBNAME)/include \
	-I.

CXX_CFLAGS := -std=c++11 \
	-fstack-protector \
	-Wall \
	-Werror \
	-Wno-sign-compare \
	-Wformat \
	-Wformat-security

LDFLAGS := -shared \
	-fstack-protector \
	-fpic \
	-L/usr/local \
	-lssl \
	-lcrypto

GAZER_LIB := $(LIBNAME)/lib$(LIBNAME).so
-include $(LIBNAME)/cc/Makefile

.PHONY: build
build: $(GAZER_LIB)
	@echo -e "\033[1;33m[BUILD] $$t \033[0m" \
        "build wheel package"
	$(PYTHON) setup.py bdist_wheel
	@ls dist/*.whl

.PHONY: test
test:
	for t in $(TESTS); do \
		@echo -e "\033[1;33m[TEST] $$t \033[0m" ; \
		$(PYTHON) $$t || exit 1; \
		echo ; \
	done

.PHONY: clean
clean:
	@rm -fr dist/
	@rm -fr build/
	@rm -fr *.egg-info/
	@find -name *.o -exec rm -fr {} \;
	@find -name *.d -exec rm -fr {} \;
	@find -name *.so -exec rm -fr {} \;

.DEFAULT_GOAL := build
