void SigQuitHandler(int signum);
int filar_map(int occ);
int filar_unmap(int occ);
unsigned int* filar_getbuffer(int occ);
int filar_setreq(int occ, int channel, int number);
int filar_init(int occ);
int filar_exit(int occ);
int filar_getbuf(int occ, int channel, int mode);
int filar_retbuf(int occ, int channel, int mode);
int filar_cardreset(int occ);
int filar_linkreset(int occ);
int filar_receive(int occ, int* filar_err);
int filar_conf(int occ);
void filar_setup(int occ);
int filar_read( int occ, int* );
int filar_read_init( int occ, int* );


