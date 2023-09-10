
#ifndef COMPUTECPP_EXPORT_H
#define COMPUTECPP_EXPORT_H

#ifdef COMPUTECPP_STATIC
#  define COMPUTECPP_EXPORT
#  define COMPUTECPP_NO_EXPORT
#else
#  ifndef COMPUTECPP_EXPORT
#    ifdef ComputeCpp_EXPORTS
        /* We are building this library */
#      define COMPUTECPP_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define COMPUTECPP_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef COMPUTECPP_NO_EXPORT
#    define COMPUTECPP_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef COMPUTECPP_DEPRECATED
#  define COMPUTECPP_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef COMPUTECPP_DEPRECATED_EXPORT
#  define COMPUTECPP_DEPRECATED_EXPORT COMPUTECPP_EXPORT COMPUTECPP_DEPRECATED
#endif

#ifndef COMPUTECPP_DEPRECATED_NO_EXPORT
#  define COMPUTECPP_DEPRECATED_NO_EXPORT COMPUTECPP_NO_EXPORT COMPUTECPP_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef COMPUTECPP_NO_DEPRECATED
#    define COMPUTECPP_NO_DEPRECATED
#  endif
#endif

#endif /* COMPUTECPP_EXPORT_H */
