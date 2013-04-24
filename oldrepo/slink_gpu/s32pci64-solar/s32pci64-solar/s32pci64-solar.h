/**************
 * Prototypes *
 **************/

void solar_reset(int);
void solar_solar(int);
int solar_map(int);
int solar_unmap(int);
int solar_init(int occ);
void solar_setup(int occ); 
int solar_cardreset(int);
int solar_linkreset(int);
unsigned int * solar_getbuffer(int occ);
int solar_send(int,int);
int solar_send_ptr(int len, unsigned* userptr, int occ);
int solar_send_ptr4(int len, unsigned* userptr, int occ, int buffer);
int solar_exit(int occ);
void solar_testmode_on(int occ); 
void solar_testmode_off(int occ);
void solar_dump_opstat(int occ); 
void solar_dump_opctrl(int occ);
int solar_req_free(int occ);
