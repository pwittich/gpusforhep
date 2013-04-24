/************************************************************************/
/*									*/
/*  This is the CMEM_RCC driver	 					*/
/*  Its purpose is to provide user applications with contiguous data 	*/
/*  buffers for DMA operations. It is not based on any extensions to 	*/
/*  the Linux kernel like the BigPhysArea patch.			*/
/*									*/
/*  12. Dec. 01  MAJO  created						*/
/*									*/
/*******C 2001 - The software with that certain something****************/

/* KH 2.6 */
#include <linux/types.h>
#include <linux/init.h>

//#include <linux/config.h>
//#include <linux/autoconf.h>
#include <generated/autoconf.h> //Wes, 03.03.11
#if defined(CONFIG_MODVERSIONS) && !defined(MODVERSIONS)
  #define MODVERSIONS
#endif
//MJ-SMP:#ifdef CONFIG_SMP
//MJ-SMP:  #define __SMP__
//MJ-SMP:#endif

#if defined(MODVERSIONS)
//#include <linux/modversions.h>
  /* KH 2.6 */
  #include <config/modversions.h>
#endif

#ifdef BPA
  #include <linux/bigphysarea.h>
#endif
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/string.h>
#include <linux/errno.h>
#include <linux/mm.h>
#include <linux/vmalloc.h>
#include <linux/mman.h>
/* KH commented for amd64     */
/* #include <linux/wrapper.h> */
#include <linux/slab.h>
#include <linux/proc_fs.h>
#include <linux/sched.h>
#include <current.h>
#include <asm/io.h>
#include <asm/uaccess.h>
#include "cmem_rcc/cmem_rcc_drv.h"

#ifdef MODULE
  MODULE_DESCRIPTION("Allocation of contiguous memory");
  MODULE_AUTHOR("Markus Joos, CERN/EP");
  #ifdef MODULE_LICENSE
    MODULE_LICENSE("Private: Contact markus.joos@cern.ch");
  #endif
#endif

// Prototypes
int cmem_rcc_drv_open(struct inode *inode, struct file *file);
int cmem_rcc_drv_release(struct inode *inode, struct file *file);
//int cmem_rcc_drv_ioctl(struct inode *inode, struct file *file, unsigned int cmd, unsigned long arg);
int cmem_rcc_drv_ioctl(struct file *file,unsigned int cmd, unsigned long arg);
int cmem_rcc_drv_mmap(struct file *file, struct vm_area_struct *vma);
void cmem_rcc_drv_vclose(struct vm_area_struct *vma);

// The ordinary device operations
static struct file_operations cmem_rcc_drv_fops =
{
/* #if LINUX_VERSION_CODE >= KERNEL_VERSION(2,4,0) */
  owner:   THIS_MODULE,
/* #endif */
  mmap:    cmem_rcc_drv_mmap,
  open:    cmem_rcc_drv_open,
  //compat_ioctl:   cmem_rcc_drv_ioctl,
  unlocked_ioctl: cmem_rcc_drv_ioctl,
  release: cmem_rcc_drv_release
};

// memory handler functions
static struct vm_operations_struct cmem_rcc_drv_vm_ops = 
{
  close:  cmem_rcc_drv_vclose,   // mmap-close
};
  
// Major number of device
static int major;

// Globals
// KH 2.6 add static 

static buffer_t *buffer_table;
#ifdef BPA
  bpa_buffer_t *bpa_buffer_table;
#endif

/*********************************************************************************************/
static int proc_read_cmem(char *page, char **start, off_t off, int count, int *eof, void *data)
/*********************************************************************************************/
{
  int loop, len;

    
  len = sprintf(page, "CMEM RCC driver\n");
    
  len += sprintf(page + len, "__get_free_pages\n");
  len += sprintf(page + len, "  PID | Phys. address |          Size | Order | Name\n");
  for(loop = 0; loop < MAX_BUFFS; loop++)
  {
    if (len + 100 > count)
    {
      len += sprintf(page + len, "Cannot display additional entries due to lack of memory\n");
      return len;
    }
    if (buffer_table[loop].size)
    {
      len += sprintf(page + len, "%5d |", buffer_table[loop].pid);
      len += sprintf(page + len, "    0x%08x |", buffer_table[loop].paddr);
      len += sprintf(page + len, "    0x%08x |", buffer_table[loop].size);
      len += sprintf(page + len, "     %d |", buffer_table[loop].order);
      len += sprintf(page + len, " %s\n", buffer_table[loop].name);
    }
  }

#ifdef BPA
  len += sprintf(page + len, "BigPhysArea\n");
  len += sprintf(page + len, "  PID | Phys. address |          Size | Name\n");
  for(loop = 0; loop < MAX_BUFFS; loop++)
  {
    if (len + 100 > count)
    {
      len += sprintf(page + len, "Cannot display additional entries due to lack of memory\n");
      return len;
    }
    if (bpa_buffer_table[loop].size)
    {
      len += sprintf(page + len, "%5d |", bpa_buffer_table[loop].pid);
      len += sprintf(page + len, "    0x%08x |", bpa_buffer_table[loop].paddr);
      len += sprintf(page + len, "    0x%08x |", bpa_buffer_table[loop].size);
      len += sprintf(page + len, " %s\n", bpa_buffer_table[loop].name);
    }
  }
#endif

  return len;
}


/************************************************************************************************/
static int cmem_rcc_drv_table_read(char *buf, char **start, off_t offset, int buf_len, int unused)
/************************************************************************************************/
{
  int eof = 0;

  return proc_read_cmem(buf, start, offset, buf_len, &eof, NULL);
}

/* KH 2.6 */
/* #if LINUX_VERSION_CODE >= KERNEL_VERSION(2,4,0) */
  static struct proc_dir_entry *cmem_rcc_file;
/* #else 
  struct proc_dir_entry cmem_rcc_drv_proc_file = 
  {
    0,
    8,
    "cmem_rcc",
    S_IFREG|S_IRUGO,
    1,
    0,
    0,
    0,
    NULL,
    &cmem_rcc_drv_table_read
  };
#endif
*/

/*******************/
static int __init cmem_init_module(void)
/*******************/
{
  int loop;

  kdebug(("cmem_rcc_drv(init_module): calling register_chrdev\n"));
  if ((major = register_chrdev(0, "cmem_rcc", &cmem_rcc_drv_fops)) < 0) 
  {
    printk("cmem_rcc_drv(init_module): unable to register character device\n");
    return (-EIO);
  }
  kdebug(("cmem_rcc_drv(init_module): major = %d\n", major));
 
  // Allocate memory for buffer table
  kdebug(("cmem_rcc_drv(init_module): MAX_BUFFS        = %d\n", MAX_BUFFS));
  kdebug(("cmem_rcc_drv(init_module): sizeof(buffer_t) = %d\n", sizeof(buffer_t)));
  kdebug(("cmem_rcc_drv(init_module): need %d bytes for __get_free_pages buffer table\n", MAX_BUFFS * sizeof(buffer_t)));
  buffer_table = (buffer_t *)kmalloc(MAX_BUFFS * sizeof(buffer_t), GFP_KERNEL);
  if (buffer_table == NULL)
  {
    printk("cmem_rcc_drv(init_module): unable to allocate memory for __get_free_pages buffer table\n");
    return (-ENOMEM);
  }
  
  // Clear the buffer table
  for(loop = 0; loop < MAX_BUFFS; loop++)
    {
    buffer_table[loop].paddr = 0;
    buffer_table[loop].size  = 0;
    buffer_table[loop].pid   = 0;
    }

#ifdef BPA    
  // Allocate memory for bpa_buffer table
  kdebug(("cmem_rcc_drv(init_module): need %d bytes for BPA buffer table\n", MAX_BUFFS * sizeof(bpa_buffer_t)));
  bpa_buffer_table = (bpa_buffer_t *)kmalloc(MAX_BUFFS * sizeof(bpa_buffer_t), GFP_KERNEL);
  if (bpa_buffer_table == NULL)
  {
    printk("cmem_rcc_drv(init_module): unable to allocate memory for BPA buffer table\n");
    return (-ENOMEM);
  }
  
  // Clear the bpa_buffer table
  for(loop = 0; loop < MAX_BUFFS; loop++)
  {
    bpa_buffer_table[loop].paddr = 0;
    bpa_buffer_table[loop].size  = 0;
    bpa_buffer_table[loop].pid   = 0;
  }  
#endif
      
  // Install /proc entry
  /* KH 2.6 */
/*   #if LINUX_VERSION_CODE >= KERNEL_VERSION(2,4,0) */
    cmem_rcc_file = create_proc_read_entry("cmem_rcc", 0444, NULL, proc_read_cmem, NULL);
    if (cmem_rcc_file == NULL)
    {
      printk("cmem_rcc_drv(init_module): error from call to create_proc_dir_entry\n");
      return (-ENOMEM);
    }
    //cmem_rcc_file->owner = THIS_MODULE;  //Wes, 03.03.11
    /*
  #else
    proc_register(&proc_root, &cmem_rcc_drv_proc_file);
  #endif
    */
  printk("cmem_rcc_drv(init_module): driver installed\n");
  return(0);
}


/***********************/
static void __exit cmem_cleanup_module(void)
/***********************/
{
  int loop,loop2;
  struct page *page_ptr;

  // Release orphaned __get_free_pages buffers
  for(loop = 0; loop < MAX_BUFFS; loop++)
  {
    if (buffer_table[loop].size > 0)
    {
      // unreserve all pages
      page_ptr = virt_to_page(buffer_table[loop].kaddr);
      
      for (loop2 = (1 << buffer_table[loop].order); loop2 > 0; loop2--, page_ptr++)
        clear_bit(PG_reserved, &page_ptr->flags);

      // free the area 
      free_pages(buffer_table[loop].kaddr, buffer_table[loop].order);
      kdebug(("cmem_rcc_drv(cleanup_module): Releasing orphaned buffer: paddr=0x%08x  size=0x%08x  name=%s\n",
      buffer_table[loop].paddr, buffer_table[loop].size, buffer_table[loop].name));
    }
  }
  
  // Return the buffer table
  kfree (buffer_table); 

#ifdef BPA
  // Release orphaned BPA buffers
  for(loop = 0; loop < MAX_BUFFS; loop++)
  {
    if (bpa_buffer_table[loop].size)
    {
      bigphysarea_free_pages((void *)bpa_buffer_table[loop].kaddr);
      kdebug(("cmem_rcc_drv(cleanup_module): Releasing orphaned BPA buffer: paddr=0x%08x  size=0x%08x  name=%s\n",
      bpa_buffer_table[loop].paddr, bpa_buffer_table[loop].size, bpa_buffer_table[loop].name));
    }
  }
  
  // Return the bpa_buffer table
  kfree (bpa_buffer_table);
#endif
  
  // Remove /proc entry
/*   #if LINUX_VERSION_CODE >= KERNEL_VERSION(2,4,0) */
    remove_proc_entry("cmem_rcc", NULL);
    /*
  #else
    proc_unregister(&proc_root, cmem_rcc_drv_proc_file.low_ino);
  #endif
    */
  // Unregister the device 
  unregister_chrdev(major, "cmem_rcc");
 
  printk("cmem_rcc_drv(cleanup_module): driver removed\n");
  return;
}


/***********************************************************/
int cmem_rcc_drv_open(struct inode *inode, struct file *file)
/***********************************************************/
{
  //MOD_INC_USE_COUNT;
  kdebug(("cmem_rcc_drv(cmem_rcc_drv_open) MOD_INC_USE_COUNT called\n"));
  return(0);
}


/**************************************************************/
int cmem_rcc_drv_release(struct inode *inode, struct file *file)
/**************************************************************/
{
  int loop, loop2;
  struct page *page_ptr;

  kdebug(("cmem_rcc_drv(cmem_rcc_drv_release): function called\n"));

  // Release orphaned __get_free_pages buffers of the current process
  for(loop = 0; loop < MAX_BUFFS; loop++)
  {
    if (buffer_table[loop].pid == current->pid)
    {
      // unreserve all pages
      page_ptr = virt_to_page(buffer_table[loop].kaddr);

      for (loop2 = (1 << buffer_table[loop].order); loop2 > 0; loop2--, page_ptr++)
        clear_bit(PG_reserved, &page_ptr->flags);

      // free the area 
      free_pages(buffer_table[loop].kaddr, buffer_table[loop].order);
      
      // clear the entry in the buffer table
      buffer_table[loop].size = 0;
      buffer_table[loop].paddr = 0;
      buffer_table[loop].pid = 0;
      buffer_table[loop].kaddr = 0;
      buffer_table[loop].order = 0;

      kdebug(("cmem_rcc_drv(cmem_rcc_drv_release): Releasing orphaned buffer of process %d: paddr=0x%08x  size=0x%08x  name=%s\n",
      current->pid, buffer_table[loop].paddr, buffer_table[loop].size, buffer_table[loop].name));
    }
  }

#ifdef BPA    
  // Release orphaned BPA buffers of the current process
  for(loop = 0; loop < MAX_BUFFS; loop++)
  {
    if (bpa_buffer_table[loop].pid == current->pid)
    {
      bigphysarea_free_pages((void *)bpa_buffer_table[loop].kaddr);
      
      // clear the entry in the buffer table
      bpa_buffer_table[loop].size = 0;
      bpa_buffer_table[loop].paddr = 0;
      bpa_buffer_table[loop].pid = 0;
      bpa_buffer_table[loop].kaddr = 0;

      kdebug(("cmem_rcc_drv(cmem_rcc_drv_release): Releasing orphaned BPA buffer of process %d: paddr=0x%08x  size=0x%08x  name=%s\n",
      current->pid, bpa_buffer_table[loop].paddr, bpa_buffer_table[loop].size, bpa_buffer_table[loop].name));
    }
  }  
#endif
    
  //MOD_DEC_USE_COUNT;
  kdebug(("cmem_rcc_drv(cmem_rcc_drv_release): MOD_DEC_USE_COUNT called\n"));
  return(0);
}



/*************************************************************************************************/
//int cmem_rcc_drv_ioctl(struct inode *inode, struct file *file, unsigned int cmd, unsigned long arg)
int cmem_rcc_drv_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
/*************************************************************************************************/
{
  kdebug(("Made it to the ioctl function: Execute %d\n",cmd));

  switch (cmd)
  {
#ifdef BPA
    case CMEM_RCC_BPA_GET:
    {
      unsigned int tnum, ok, pagecount;
      cmem_rcc_t uio_desc;
      
      if (copy_from_user(&uio_desc, (void *)arg, sizeof(cmem_rcc_t)) !=0)
      {
        printk("cmem_rcc_drv(ioctl,CMEM_RCC_GET): error in from copy_from_user\n");
        return(-EFAULT);
      }   
      kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_BPA_GET): uio_desc.size = 0x%08x\n", uio_desc.size)); 

//MJ-SMP: protect this fragment (preferrably with a spinlock)
      // Look for a slot in the buffer table
      ok = 0;
      for(tnum = 0; tnum < MAX_BUFFS; tnum++)
      {
        if (bpa_buffer_table[tnum].size == 0)
        {
//MJ-SMP:          bpa_buffer_table[tnum].size = 1;  //This is to reserve the entry. The real size will be added below
          ok = 1;
          kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_BPA_GET): tnum = %d\n", tnum));
          break;
        }
      }
//MJ-SMP: end of protected zone

      if (!ok)
      {
        printk("cmem_rcc_drv(ioctl,CMEM_RCC_GET): all buffers are in use\n");
        return -ENOMEM;
      }
     
      pagecount = (int)((uio_desc.size - 1) / PAGE_SIZE + 1); // pages
      kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_BPA_GET): requested number of pages = %d\n", pagecount));

      uio_desc.kaddr = (int)bigphysarea_alloc_pages(pagecount, 0, GFP_KERNEL);
      if (uio_desc.kaddr == 0) 
      {
       printk("cmem_rcc_drv(ioctl,CMEM_RCC_BPA_GET): error on bigphysarea_alloc_pages\n"); 
//MJ-SMP: bpa_buffer_table[tnum].size = 0;  //Not required any more
       return -EFAULT;
      } 
      
      uio_desc.paddr = virt_to_bus((void *) uio_desc.kaddr);
      uio_desc.size = PAGE_SIZE * pagecount;

      // Put an entry into the buffer table
      bpa_buffer_table[tnum].size = uio_desc.size;
      bpa_buffer_table[tnum].paddr = uio_desc.paddr;
      bpa_buffer_table[tnum].pid = current->pid;
      bpa_buffer_table[tnum].kaddr = uio_desc.kaddr;
      strcpy(bpa_buffer_table[tnum].name, uio_desc.name);
  
      kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_BPA_GET): PAGE_SIZE       = 0x%08x\n", (unsigned int)PAGE_SIZE));
      kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_BPA_GET): uio_desc.kaddr  = 0x%08x\n", (unsigned int)uio_desc.kaddr));
      kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_BPA_GET): uio_desc.paddr  = 0x%08x\n", (unsigned int)uio_desc.paddr));
      kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_BPA_GET): uio_desc.size   = 0x%08x\n", (unsigned int)uio_desc.size));
      kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_BPA_GET): uio_desc.name   = %s\n", uio_desc.name));

      if (copy_to_user((void *)arg, &uio_desc, sizeof(cmem_rcc_t)) != 0)
      {
        printk("cmem_rcc_drv(ioctl,CMEM_RCC_GET): error in from copy_to_user\n");
        return(-EFAULT);
      }

      kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_BPA_GET):bpa_buffer_table[%d].size = 0x%08x\n", tnum, bpa_buffer_table[tnum].size)); 
      return 0;
      break;
    }    
    
    case CMEM_RCC_BPA_FREE:
    {
      //Release memory
      unsigned int ok, tnum;
      cmem_rcc_t uio_desc;

      if (copy_from_user(&uio_desc, (void *)arg, sizeof(cmem_rcc_t)) !=0 )
      {
        printk("cmem_rcc_drv(ioctl,CMEM_RCC_BPA_FREE): error in from copy_from_user\n");
        return(-EFAULT);
      } 
         
      kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_BPA_FREE): uio_desc.size = 0x%08x\n", uio_desc.size));
      kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_BPA_FREE): uio_desc.paddr = 0x%08x\n", uio_desc.paddr));
 
      // Look for the buffer in the buffer table
      ok = 0;
      for(tnum = 0; tnum < MAX_BUFFS; tnum++)
      {
        kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_BPA_FREE): bpa_buffer_table[%d].size = 0x%08x\n", tnum, bpa_buffer_table[tnum].size));
        kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_BPA_FREE): bpa_buffer_table[%d].paddr = 0x%08x\n", tnum, bpa_buffer_table[tnum].paddr));
        if (bpa_buffer_table[tnum].size == uio_desc.size  &&  bpa_buffer_table[tnum].paddr == uio_desc.paddr)
        {
          ok = 1;
          kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_BPA_FREE): tnum = %d\n", tnum));
          break;
        }
      }
      
      if (!ok)
      {
        printk("cmem_rcc_drv(ioctl,CMEM_RCC_BPA_FREE): Failed to find buffer in table\n");
        return -EINVAL;
      }
      
      bigphysarea_free_pages((void *)bpa_buffer_table[tnum].kaddr);
      kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_BPA_FREE): memory freed @ address 0x%8x\n", bpa_buffer_table[tnum].kaddr));

      // Delete the entry in the buffer table
      bpa_buffer_table[tnum].paddr = 0;
      bpa_buffer_table[tnum].pid = 0;
      bpa_buffer_table[tnum].kaddr = 0;
      bpa_buffer_table[tnum].size = 0;  //This enables the entry to be re-used
      return 0;
      break;
    }
#endif
    
    case CMEM_RCC_GET:
    {
      // Allocate memory
      unsigned int loop, tnum, ok;
      cmem_rcc_t uio_desc;
      struct page *page_ptr;

      if (copy_from_user(&uio_desc, (void *)arg, sizeof(cmem_rcc_t)) !=0 )
      {
        printk("cmem_rcc_drv(ioctl,CMEM_RCC_GET): error in from copy_from_user\n");
        return(-EFAULT);
      }   
      kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_GET): uio_desc.order = %d\n", uio_desc.order));

//MJ-SMP: protect this fragment (preferrably with a spinlock)
      // Look for a slot in the buffer table
      ok = 0;
      for(tnum = 0; tnum < MAX_BUFFS; tnum++)
      {
        if (buffer_table[tnum].size == 0)
        {
//MJ-SMP:         buffer_table[tnum].size = 1;  //This is to reserve the entry. The real size will be added below
          ok = 1;
          kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_GET): tnum = %d\n", tnum));
          break;
        }
      }
//MJ-SMP: end of protected zone

      if (!ok)
      {
        printk("cmem_rcc_drv(ioctl,CMEM_RCC_GET): all buffers are in use\n");
        return -ENOMEM;
      }

      uio_desc.kaddr = 0;
      // Get a physically contigous memory area with __get_free_pages 
      kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_GET): calling __get_free_pages\n"));
      uio_desc.kaddr = __get_free_pages(GFP_ATOMIC, uio_desc.order);
      kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_GET): __get_free_pages returns address 0x%08x\n", (unsigned int)uio_desc.kaddr));
      if (!uio_desc.kaddr)
      {
        printk("cmem_rcc_drv(ioctl,CMEM_RCC_GET): error from __get_free_pages for order=%d\n", uio_desc.order);
//MJ-SMP:         buffer_table[tnum].size = 0;  // No longer required
        return(-EFAULT);
      }

      // Reserve all pages to make them remapable
/*       #if LINUX_VERSION_CODE < KERNEL_VERSION(2,4,0) */
/*         page_ptr = mem_map + MAP_NR(uio_desc.kaddr); */
/*       #else */
        page_ptr = virt_to_page(uio_desc.kaddr);
/*       #endif */

      kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_GET): reserving pages...\n")); 
      for (loop = (1 << uio_desc.order); loop > 0; loop--, page_ptr++)
        set_bit(PG_reserved, &page_ptr->flags);
      kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_GET): ... done\n"));
  
      uio_desc.paddr = virt_to_bus((void *) uio_desc.kaddr);
      uio_desc.size = PAGE_SIZE * (1 << uio_desc.order);

      kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_GET): PAGE_SIZE       = 0x%08x\n", (unsigned int)PAGE_SIZE));
      kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_GET): uio_desc.kaddr  = 0x%08x\n", (unsigned int)uio_desc.kaddr));
      kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_GET): uio_desc.paddr  = 0x%08x\n", (unsigned int)uio_desc.paddr));
      kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_GET): uio_desc.size   = 0x%08x\n", (unsigned int)uio_desc.size));
      kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_GET): uio_desc.order  = 0x%08x\n", (unsigned int)uio_desc.order));

      // Put an entry into the buffer table
      buffer_table[tnum].size = uio_desc.size;
      buffer_table[tnum].paddr = uio_desc.paddr;
      buffer_table[tnum].pid = current->pid;
      buffer_table[tnum].kaddr = uio_desc.kaddr;
      buffer_table[tnum].order = uio_desc.order;
      strcpy(buffer_table[tnum].name, uio_desc.name);
  
      kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_GET): uio_desc.name           = %s\n", uio_desc.name));
      kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_GET): buffer_table[tnum].name = %s\n", buffer_table[tnum].name));

      if (copy_to_user((void *)arg, &uio_desc, sizeof(cmem_rcc_t)) != 0)
      {
        printk("cmem_rcc_drv(ioctl,CMEM_RCC_GET): error in from copy_to_user\n");
        return(-EFAULT);
      }
      kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_GET): end of function\n"));
      break;
    }
    
    case CMEM_RCC_FREE:
    {
      //Release memory
      unsigned int loop, ok, tnum;
      cmem_rcc_t uio_desc;
      struct page *page_ptr;

      if (copy_from_user(&uio_desc, (void *)arg, sizeof(cmem_rcc_t)) !=0 )
      {
        printk("cmem_rcc_drv(ioctl,CMEM_RCC_FREE): error in from copy_from_user\n");
        return(-EFAULT);
      } 
      kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_FREE): uio_desc.order = %d\n", uio_desc.order));
      kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_FREE): uio_desc.kaddr = 0x%08x\n", uio_desc.kaddr));
    
      // Look for the buffer in the buffer table
      ok = 0;
      for(tnum = 0; tnum < MAX_BUFFS; tnum++)
      {
        if (buffer_table[tnum].size == uio_desc.size  &&  buffer_table[tnum].paddr == uio_desc.paddr)
        {
          ok = 1;
          kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_FREE): buffer_table[%d] matches\n", tnum));
          break;
        }
      }
      
      if (!ok)
      {
        printk("cmem_rcc_drv(ioctl,CMEM_RCC_FREE): Failed to find buffer in table\n");
        return -EINVAL;
      }
      
      // unreserve all pages
      kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_FREE): unreserving pages ...\n"));
/*       #if LINUX_VERSION_CODE < KERNEL_VERSION(2,4,0) */
/*         page_ptr = mem_map + MAP_NR(uio_desc.kaddr); */
/*       #else */
        page_ptr = virt_to_page(uio_desc.kaddr);
/*       #endif */

      for (loop = (1 << uio_desc.order); loop > 0; loop--, page_ptr++)
        clear_bit(PG_reserved, &page_ptr->flags);
      kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_FREE): ... done\n"));

      // free the area 
      free_pages(uio_desc.kaddr, uio_desc.order);
      kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_FREE): pages freed\n")); 
      // Delete the entry in the buffer table
      buffer_table[tnum].paddr = 0;
      buffer_table[tnum].pid = 0;
      buffer_table[tnum].kaddr = 0;
      buffer_table[tnum].order = 0;
      buffer_table[tnum].size = 0;   //This enables the entry to be re-used
      kdebug(("cmem_rcc_drv(ioctl,CMEM_RCC_FREE): buffer_table[%d] cleared\n", tnum));
      break; 
    }
       
  case DUMP:
    {
      char *buf;
      int len, loop;

      kdebug(("cmem_rcc_drv(ioctl,DUMP): called\n"));

      buf = (char *)kmalloc(TEXT_SIZE, GFP_KERNEL);
      if (buf == NULL)
      {
        kdebug(("cmem_rcc_drv(ioctl,DUMP): error from kmalloc\n"));
        return(-EFAULT);
      }

      len = 0;
      len += sprintf(buf + len, "Memory allocated by __get_free_pages\n");
      len += sprintf(buf + len, "  PID | Phys. address |          Size | Order | Name\n");
      for(loop = 0; loop < MAX_BUFFS; loop++)
      {
        if (buffer_table[loop].size)
        {
          len += sprintf(buf + len, "%5d |", buffer_table[loop].pid);
          len += sprintf(buf + len, "    0x%08x |", buffer_table[loop].paddr);
          len += sprintf(buf + len, "    0x%08x |", buffer_table[loop].size);
          len += sprintf(buf + len, "     %d |", buffer_table[loop].order);
          len += sprintf(buf + len, " %s\n", buffer_table[loop].name);
        }
      }

#ifdef BPA
      len += sprintf(buf + len, "Memory allocated by BigPhysArea\n");
      len += sprintf(buf + len, "  PID | Phys. address |          Size | Name\n");
      for(loop = 0; loop < MAX_BUFFS; loop++)
      {
        if (bpa_buffer_table[loop].size)
        {
          len += sprintf(buf + len, "%5d |", bpa_buffer_table[loop].pid);
          len += sprintf(buf + len, "    0x%08x |", bpa_buffer_table[loop].paddr);
          len += sprintf(buf + len, "    0x%08x |", bpa_buffer_table[loop].size);
          len += sprintf(buf + len, " %s\n", bpa_buffer_table[loop].name);
        }
      }
#endif

      if (copy_to_user((void *)arg, buf, TEXT_SIZE * sizeof(char)) != 0)
      {
	kdebug(("cmem_rcc_drv(ioctl,DUMP): error from copy_to_user\n"));
	return(-EFAULT);
      }
            
      kfree(buf);
      break;
    }
  }
  return(0);
}


// 2.4.x: this method is called from do_mmap_pgoff, from
// do_mmap, from the syscall. The caller of do_mmap grabs
// the mm semaphore. So we are protected from races here.
/******************************************************************/
int cmem_rcc_drv_mmap(struct file *file, struct vm_area_struct *vma)
/******************************************************************/
{
  unsigned long offset, size;

  kdebug(("cmem_rcc_drv(cmem_rcc_drv_mmap): cmem_rcc_drv_mmap called\n"));
  
/*   #if LINUX_VERSION_CODE < KERNEL_VERSION(2,4,0) */
/*     offset = vma->vm_offset; */
/*   #else */
    offset = vma->vm_pgoff << PAGE_SHIFT;
/*   #endif */
  
  size = vma->vm_end - vma->vm_start;

  kdebug(("cmem_rcc_drv(cmem_rcc_drv_mmap): offset = 0x%08x\n", (unsigned int)offset));
  kdebug(("cmem_rcc_drv(cmem_rcc_drv_mmap): size   = 0x%08x\n", (unsigned int)size));

  if (offset & ~PAGE_MASK)
  {
    printk("cmem_rcc_drv(cmem_rcc_drv_mmap): offset not aligned: %ld\n", offset);
    return -ENXIO;
  }

  // we only support shared mappings. "Copy on write" mappings are
  // rejected here. A shared mapping that is writeable must have the
  // shared flag set.
  if ((vma->vm_flags & VM_WRITE) && !(vma->vm_flags & VM_SHARED))
  {
    printk("cmem_rcc_drv(cmem_rcc_drv_mmap): writeable mappings must be shared, rejecting\n");
    return(-EINVAL);
  }

/*   #if LINUX_VERSION_CODE >= KERNEL_VERSION(2,4,0) */
    vma->vm_flags |= VM_RESERVED;
/*   #endif */
  
  // we do not want to have this area swapped out, lock it
  vma->vm_flags |= VM_LOCKED;

  // we create a mapping between the physical pages and the virtual
  // addresses of the application with remap_page_range.
  // enter pages into mapping of application
  kdebug(("cmem_rcc_drv(cmem_rcc_drv_mmap): Parameters of remap_page_range()\n"));
  kdebug(("cmem_rcc_drv(cmem_rcc_drv_mmap): Virtual address  = 0x%08x\n", (unsigned int)vma->vm_start));
  kdebug(("cmem_rcc_drv(cmem_rcc_drv_mmap): Physical address = 0x%08x\n", (unsigned int)offset));
  kdebug(("cmem_rcc_drv(cmem_rcc_drv_mmap): Size             = 0x%08x\n", (unsigned int)size));

  /* KH 2.6 */
  /*   if (remap_page_range(vma->vm_start, offset, size, vma->vm_page_prot)) */
  /* KH 2.6.13 */
    /*   if (remap_page_range(vma, vma->vm_start, offset, size, vma->vm_page_prot)) */
  if (remap_pfn_range(vma, vma->vm_start, offset>>PAGE_SHIFT, size, vma->vm_page_prot))
  {
    printk("cmem_rcc_drv(cmem_rcc_drv_mmap): remap page range failed\n");
    return -ENXIO;
  }
  
  vma->vm_ops = &cmem_rcc_drv_vm_ops;  
  //MOD_INC_USE_COUNT;
  kdebug(("cmem_rcc_drv(cmem_rcc_drv_mmap) MOD_INC_USE_COUNT called\n"));
  
  kdebug(("cmem_rcc_drv(cmem_rcc_drv_mmap): cmem_rcc_drv_mmap done\n"));
  return(0);
}


/*************************************************/
void cmem_rcc_drv_vclose(struct vm_area_struct *vma)
/*************************************************/
{  
  kdebug(("cmem_rcc_drv(cmem_rcc_drv_vclose): Virtual address  = 0x%08x\n", (unsigned int)vma->vm_start));
  kdebug(("cmem_rcc_drv(cmem_rcc_drv_vclose): mmap released\n"));
  //MOD_DEC_USE_COUNT;
  kdebug(("cmem_rcc_drv(cmem_rcc_drv_vclose): MOD_DEC_USE_COUNT called\n"));
}



module_init(cmem_init_module);
module_exit(cmem_cleanup_module);

/* module_init(cmem_init); */
/* module_exit(cmem_cleanup); */
