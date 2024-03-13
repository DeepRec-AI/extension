CXX ?= gcc

ifeq ($(PYTHON_VERSION), 2)
$(warning using python version 2)
PYTHON ?= python2
else
$(warning default using python version 3)
PYTHON ?= python3
endif

CFLAGS := -O3 -g \
	-DNDEBUG \
	-Iinclude \
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
	-fpic


GAZER_LIB := gazer/libgazer.so
include gazer/cc/Makefile

.PHONY: gazer
gazer: $(GAZER_LIB)
	@echo -e "\033[1;33m[BUILD] $$t \033[0m" \
        "build wheel package"
	cd gazer/; $(PYTHON) setup.py bdist_wheel
	@ls gazer/dist/*.whl

.PHONY: clean
clean:
	@rm -fr gazer/dist/
	@rm -fr gazer/build/
	@rm -fr gazer/*.egg-info/
	@rm -fr third_party/rapidjson/build
	@rm -fr third_party/grpc/build
	@rm -fr third_party/protobuf/build
	@echo "remove .o/.d/.so/.pb*"
	@find ./ -name *.pb.* -exec rm -rf {} \;
	@find -name *.o -exec rm -fr {} \;
	@find -name *.d -exec rm -fr {} \;
	@find -name *.so -exec rm -fr {} \;

.DEFAULT_GOAL := gazer
