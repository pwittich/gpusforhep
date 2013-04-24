#############################################################

# override compiler
CC   = gcc

#############################################################

LOCAL_ROOT_DIR= $(PWD)
LOCAL_LIB = $(LOCAL_ROOT_DIR)/lib/libl2slink.a
PACKAGE_LIB   = $(subst $(LOCAL_ROOT_DIR), $(ROOT_DIR), $(LOCAL_LIB) )

DIRS =  $(LOCAL_ROOT_DIR)/io_rcc \
	$(LOCAL_ROOT_DIR)/cmem_rcc \
	$(LOCAL_ROOT_DIR)/rcc_time_stamp \
	$(LOCAL_ROOT_DIR)/rcc_error \
        $(LOCAL_ROOT_DIR)/s32pci64-filar \
	$(LOCAL_ROOT_DIR)/s32pci64-solar 


IFLAGS= $(foreach DIR, $(DIRS), -I$(DIR) )
CFLAGS += $(CCFLAGS) $(IFLAGS)

export LOCAL_ROOT_DIR LOCAL_LIB CFLAGS IFLAGS MAKE

#############################################################

% : 
	@for i in $(DIRS); \
	do cd $$i/src && $(MAKE) $@; \
	done; \
	rm -f $(PACKAGE_LIB);\
	if [ ! $@ == clean ]; then \
	ln -s $(LOCAL_LIB) $(PACKAGE_LIB); \
	else 	rm -f $(LOCAL_LIB); \
	fi



