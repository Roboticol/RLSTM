TARGET_EXEC := main

BUILD_DIR := ./build
SRC_DIR := ./src

SRCS := $(shell find $(SRC_DIR) -name '*.c')
OBJS := $(SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

all : $(BUILD_DIR)/$(TARGET_EXEC)

$(BUILD_DIR)/$(TARGET_EXEC) : $(OBJS)
	gcc $^ -o $@ -lm -lgsl

$(BUILD_DIR)/%.o : $(SRC_DIR)/%.c
	gcc -c $< -o $@

.PHONY : clean

clean :
	rm -rf $(BUILD_DIR) && mkdir $(BUILD_DIR) 
