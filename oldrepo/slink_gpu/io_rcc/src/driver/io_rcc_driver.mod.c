#include <linux/module.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

MODULE_INFO(vermagic, VERMAGIC_STRING);

struct module __this_module
__attribute__((section(".gnu.linkonce.this_module"))) = {
 .name = KBUILD_MODNAME,
 .init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
 .exit = cleanup_module,
#endif
 .arch = MODULE_ARCH_INIT,
};

static const struct modversion_info ____versions[]
__used
__attribute__((section("__versions"))) = {
	{ 0x539e6e35, "module_layout" },
	{ 0x3c2c5af5, "sprintf" },
	{ 0xd509a9e5, "create_proc_entry" },
	{ 0x3d4f86e4, "__register_chrdev" },
	{ 0x9e066240, "pci_bus_write_config_dword" },
	{ 0xafa05845, "pci_bus_read_config_byte" },
	{ 0xe86f4e53, "pci_bus_read_config_word" },
	{ 0x8ac23088, "pci_bus_read_config_dword" },
	{ 0x8a59efda, "pci_dev_put" },
	{ 0x78a813df, "pci_get_device" },
	{ 0x2da418b5, "copy_to_user" },
	{ 0x33d169c9, "_copy_from_user" },
	{ 0x1dd4e4ab, "remap_pfn_range" },
	{ 0x8e95e407, "remove_proc_entry" },
	{ 0x6bc3fbc0, "__unregister_chrdev" },
	{ 0xb72397d5, "printk" },
};

static const char __module_depends[]
__used
__attribute__((section(".modinfo"))) =
"depends=";

