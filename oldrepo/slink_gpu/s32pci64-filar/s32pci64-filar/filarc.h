void SigQuitHandler(int signum);
int filar_map(int occ);
int filar_unmap(void);
int setreq(int channel, int number);
int uio_init(void);
int uio_exit(void);
int getbuf(int channel, int mode);
int retbuf(int channel, int mode);
int cardreset(void);
int linkreset(void);
int receive(int* filar_err);
int filarconf(void);
void setup(void);
int filar_read( int* );
int filar_read_init( int* );


