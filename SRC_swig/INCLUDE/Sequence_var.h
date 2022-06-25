#include "LKH.h"

Node **t;       /* The sequence of nodes to be used in a move */
Node **T;       /* The currently best t's */
Node **tSaved;  /* For saving t when using the BacktrackKOptMove function */
int *p;         /* The permutation corresponding to the sequence in which
                   the t's occur on the tour */
int *q;         /* The inverse permutation of p */
int *incl;      /* Array: incl[i] == j, if (t[i], t[j]) is an inclusion edge */
int *cycle;     /* Array: cycle[i] is cycle number of t[i] */
GainType *G;    /* For storing the G-values in the BestKOptMove function */
int K;          /* The value K for the current K-opt move */