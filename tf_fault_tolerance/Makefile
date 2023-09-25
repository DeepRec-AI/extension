TFT_SRC_DIR := $(TFT)/tf_fault_tolerance

TFT_CC_SOURCES := $(shell \
	find $(TFT_SRC_DIR)/cc -type f \
	\( -name "*.cc" ! -name "*.cu*" \) \
	-exec realpath {} --relative-to . \;)

TFT_OBJS := $(TFT_CC_SOURCES:.cc=.o)

TFT_LIB := $(TFT_SRC_DIR)/lib$(TFT).so

$(TFT_OBJS): %.o:%.cc
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) $(CXX_CFLAGS) $(TENSORFLOW_CFLAGS) -I$(TFT) \
	-MP -MMD -MF $<.d -o $@ -c $< -fpic

$(TFT_LIB): $(TFT_OBJS)
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) -std=c++11 -Wl,--version-script=$(TFT)/tft.script \
	-o $@ $(TFT_OBJS) $(TENSORFLOW_LDFLAGS) $(LDFLAGS)
