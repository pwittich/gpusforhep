typedef struct {
  __u32 ack;
  __u32 scw;
  __u32 ecw;
  int loops;
} sfio_err_info_t;

enum {
  SFIO_ERRGET,
  SFIO_CARDSETUP
};
