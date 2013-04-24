typedef struct {
  unsigned int ocr;		/*0x000 */
  unsigned int osr;		/*0x004 */
  unsigned int imask;		/*0x008 */
  unsigned int fifostat;	/*0x00c */
  unsigned int scw;		/*0x010 */
  unsigned int ecw;		/*0x014 */
  unsigned int testin;		/*0x018 */
  unsigned int d1[57];		/*0x01c-0x0fc */
  unsigned int req1;		/*0x100 */
  unsigned int ack1;		/*0x104 */
  unsigned int d2[2];		/*0x108-0x10c */
  unsigned int req2;		/*0x110 */
  unsigned int ack2;		/*0x114 */
  unsigned int d3[2];		/*0x118-0x11c */
  unsigned int req3;		/*0x120 */
  unsigned int ack3;		/*0x124 */
  unsigned int d4[2];		/*0x128-0x12c */
  unsigned int req4;		/*0x130 */
  unsigned int ack4;		/*0x134 */
  unsigned int d5[2];		/*0x138-0x13c */
} T_filar_regs;

unsigned int pcidefault[16] = {
  0x001410dc, 0x00800000, 0x02800000, 0x0000ff00, 0xfffffc00, 0x00000000,
  0x00000000, 0x00000000,
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
  0x00000000, 0x000001ff
};
