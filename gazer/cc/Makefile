SRC_DIR := gazer/cc/

TENSORFLOW_VERSION := $(shell \
	    LD_LIBRARY_PATH=/usr/local/gcc-5.3.0/lib64:/usr/local/lib64/ $(PYTHON) -c \
	    "import tensorflow as tf; print(tf.__version__)" 2>/dev/null)

TENSORFLOW_CFLAGS := $(shell \
        $(PYTHON) -c \
        "import tensorflow as tf; cflags=tf.sysconfig.get_compile_flags(); print(' '.join([c.replace('-I', '-isystem ', 1) + ' ' + c.replace('-I', '-isystem ', 1) + '/external/sparsehash_c11' + ' ' + c.replace('-I', '-isystem ', 1) + '/external/com_github_google_leveldb/include/' + ' ' + c.replace('-I', '-isystem ', 1) + '/external/libcuckoo/' if c.startswith('-I') else c for c in cflags]))" 2>/dev/null)
TENSORFLOW_LDFLAGS := \
        -Wl,-rpath='$$ORIGIN/..:$$ORIGIN/../tensorflow' \
        $(shell \
        $(PYTHON) -c \
        "import tensorflow as tf; ldflags=tf.sysconfig.get_link_flags(); print(' '.join(ldflags))" 2>/dev/null)


$(warning "Tensorflow Version" $(TENSORFLOW_VERSION))
$(warning "CFLAGS" $(TENSORFLOW_CFLAGS))
$(warning "LDFLAGS" $(TENSORFLOW_LDFLAGS))

CC_SOURCES := $(shell \
        find $(SRC_DIR) -type f \
        \( -name "*.cc" ! -name "*.cu*" \) \
        -exec realpath {} --relative-to . \;)

OBJS := $(CC_SOURCES:.cc=.o)

$(OBJS): %.o:%.cc
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) $(TENSORFLOW_CFLAGS) $(CXX_CFLAGS) \
        -MMD -MP -MF $<.d -o $@ -c $< -fpic
	sed -i '/site-packages/d' $<.d
	sed -i '/^$$/N;/^\n$$/D' $<.d

$(GAZER_LIB): $(OBJS)
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) -std=c++11 \
        -o $@ $(OBJS) \
        $(LDFLAGS) \
        $(TENSORFLOW_LDFLAGS)

