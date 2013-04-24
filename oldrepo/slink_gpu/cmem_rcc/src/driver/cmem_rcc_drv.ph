#ifndef __cmem_rcc_drv_ph__
#define __cmem_rcc_drv_ph__
/*
#ifdef __cplusplus
extern "C" {
#endif
*/

__init int cmem_init_module (void);
__exit void cmem_cleanup_module (void);
int cmem_rcc_drv_open (struct inode *inode, struct file *file);
int cmem_rcc_drv_release (struct inode *inode, struct file *file);
int cmem_rcc_drv_ioctl (struct inode *inode, struct file *file, unsigned int cmd, unsigned long arg);
int cmem_rcc_drv_mmap (struct file *file, struct vm_area_struct *vma);
void cmem_rcc_drv_vclose (struct vm_area_struct *vma);

/*
#ifdef __cplusplus
}
#endif
*/
#endif

