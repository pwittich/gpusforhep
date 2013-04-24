typedef struct {
  unsigned int opctrl;         /*0x000*/
  unsigned int opstat;         /*0x004*/
  unsigned int intmask;        /*0x008*/
  unsigned int slidasctrl;     /*0x00c*/
  unsigned int opfeat;         /*0x010*/
  unsigned int wordcnt;        /*0x014*/
  unsigned int bctrlw;         /*0x018*/
  unsigned int ectrlw;         /*0x01c*/
  unsigned int reserved[56];   /*0x020 - 0x0fc*/
  unsigned int reqfifo1;       /*0x100*/               /* PCI address FIFO */
  unsigned int reqfifo2;       /*0x104*/               /* Size (in words) FIFO */
} T_solar_regs;

