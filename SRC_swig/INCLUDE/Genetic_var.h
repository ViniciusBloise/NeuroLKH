#include "Genetic.h"

int MaxPopulationSize; /* The maximum size of the population */ 
int PopulationSize;    /* The current size of the population */

CrossoverFunction Crossover;

int **Population;      /* Array of individuals (solution tours) */
GainType *PenaltyFitness;  /* The f itnesslty  (tour penalty) of each
i                             individual */
GainType *Fitness;     /* The fitness (tour cost) of each individual */