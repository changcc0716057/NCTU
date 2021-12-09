/*
 * Example code of FreeBSD kernel module
 * 
 * Homework 4, System administration 2021
 * Department of Computer Science, National Yang Ming Chiao Tung University
 * 
 * Modified by stchang@cs.nycu.edu.tw
 * This code is from https://docs.freebsd.org/en/books/arch-handbook/driverbasics/
 * 
 * The following is the original file header:
 * 
 * Simple Echo pseudo-device KLD
 *
 * Murray Stokely
 * Søren (Xride) Straarup
 * Eitan Adler
 */

#include <sys/types.h>
#include <sys/module.h>
#include <sys/systm.h>  /* uprintf */
#include <sys/param.h>  /* defines used in kernel.h */
#include <sys/kernel.h> /* types used in module initialization */
#include <sys/conf.h>   /* cdevsw struct */
#include <sys/uio.h>    /* uio struct */
#include <sys/malloc.h>
#include <sys/libkern.h>

#include <sys/sysproto.h>
#include <sys/sysctl.h>

#define BUFFERSIZE 255

/* Function prototypes */
static d_open_t      sockn_open;
static d_close_t     sockn_close;
static d_read_t      sockn_read;
static d_write_t     sockn_write;

/* Character device entry points */
static struct cdevsw sockn_cdevsw = {
        .d_version = D_VERSION,
        .d_open = sockn_open,
        .d_close = sockn_close,
        .d_read = sockn_read,
        .d_write = sockn_write,
        .d_name = "sockn",
};

struct s_sockn {
        char msg[BUFFERSIZE + 1];
        int len;
};

/* vars */
static struct cdev *sockn_dev;
static struct s_sockn *socknmsg;
static struct s_sockn *sockncount;
static bool inserted;

MALLOC_DECLARE(M_SOCKNBUF);
MALLOC_DEFINE(M_SOCKNBUF, "socknbuffer", "buffer for sockn module");
MALLOC_DECLARE(M_SOCKNCBUF);
MALLOC_DEFINE(M_SOCKNCBUF, "sockncbuffer", "buffer for sockn module(c)");

/*
 * This function is called by the kld[un]load(2) system calls to
 * determine what actions to take when a module is loaded or unloaded.
 */
static int
sockn_loader(struct module *m __unused, int what, void *arg __unused)
{
        int error = 0;

        switch (what) {
        case MOD_LOAD:                /* kldload */
                error = make_dev_p(MAKEDEV_CHECKNAME | MAKEDEV_WAITOK,
                    &sockn_dev,
                    &sockn_cdevsw,
                    0,
                    UID_ROOT,
                    GID_WHEEL,
                    0600,
                    "sockn");
                if (error != 0)
                        break;

        sockncount = malloc(sizeof(*sockncount), M_SOCKNCBUF, M_WAITOK |
                    M_ZERO);
                socknmsg = malloc(sizeof(*socknmsg), M_SOCKNBUF, M_WAITOK |
                    M_ZERO);
                printf("sockn device loaded.\n");
                break;
        case MOD_UNLOAD:
                destroy_dev(sockn_dev);
                free(sockncount, M_SOCKNCBUF);
        free(socknmsg, M_SOCKNBUF);
                printf("sockn device unloaded.\n");
                break;
        default:
                error = EOPNOTSUPP;
                break;
        }
        return (error);
}

static int
sockn_open(struct cdev *dev __unused, int oflags __unused, int devtype __unused,
    struct thread *td __unused)
{
        int error = 0;

    size_t rv;

    u_int oldval[8];
    size_t len = sizeof(oldval);

    error = kernel_sysctlbyname (td, "vm.uma.socket.stats.current", (void *)oldval, &len, NULL, 0, &rv, 0);
    if (error) return (error);

    for (int i = 0; i < BUFFERSIZE; i++) sockncount->msg[i] = 0;
    sprintf(sockncount->msg, "%u\n", oldval[0]);

    sockncount->len = strlen(sockncount->msg);

    inserted = 0;

    return (error);
}

static int
sockn_close(struct cdev *dev __unused, int fflag __unused, int devtype __unused,
    struct thread *td __unused)
{
        // uprintf("sockn closed.\n");
        return (0);
}

/*
 * The read function just takes the buf that was saved via
 * sockn_write() and returns it to userland for accessing.
 * uio(9)
 */
static int
sockn_read(struct cdev *dev __unused, struct uio *uio, int ioflag __unused)
{
        size_t amt;
        int error = 0;

    if (!inserted && socknmsg->len + sockncount->len < BUFFERSIZE) {
        for (int i = 0; i < socknmsg->len; i++) {
            sockncount->msg[sockncount->len + i] = socknmsg->msg[i];
        }
        sockncount->len += socknmsg->len - 1;
        sockncount->msg[sockncount->len + 1] = 0;
        inserted = 1;
    }

    /*
         * How big is this read operation?  Either as big as the user wants,
         * or as big as the remaining data.  Note that the 'len' does not
         * include the trailing null character.
         */

    amt = MIN(uio->uio_resid, uio->uio_offset >= sockncount->len + 1 ? 0 :
            sockncount->len + 1 - uio->uio_offset);

    if ((error = uiomove(sockncount->msg, amt, uio)) != 0)
                uprintf("uiomove failed!\n");

        return (error);
}

/*
 * sockn_write takes in a character string and saves it
 * to buf for later accessing.
 */
static int
sockn_write(struct cdev *dev __unused, struct uio *uio, int ioflag __unused)
{
        size_t amt;
        int error;

        /*
         * We either write from the beginning or are appending -- do
         * not allow random access.
         */
        if (uio->uio_offset != 0 && (uio->uio_offset != socknmsg->len))
                return (EINVAL);

        /* This is a new message, reset length */
        if (uio->uio_offset == 0)
                socknmsg->len = 0;

        /* Copy the string in from user memory to kernel memory */
        amt = MIN(uio->uio_resid, (BUFFERSIZE - socknmsg->len));

        error = uiomove(socknmsg->msg + uio->uio_offset, amt, uio);

        /* Now we need to null terminate and record the length */
        socknmsg->len = uio->uio_offset;
        socknmsg->msg[socknmsg->len] = 0;

        if (error != 0)
                uprintf("Write failed: bad address!\n");
        return (error);
}

DEV_MODULE(sockn, sockn_loader, NULL);
