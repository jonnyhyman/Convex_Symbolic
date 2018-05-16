
#ifndef PROBLEM_H
#define PROBLEM_H

#include "ecos.h"


double A[30][20];
double b_[30][1];



idxint q[2];


pfloat c[82];
pfloat h[74];
pfloat Gpr[74];
pfloat b[60];
pfloat Apr[690];


idxint Gir[74];
idxint Gjc[83];
idxint Air[690];
idxint Ajc[83];



void gather_matrices();

#endif //PROBLEM_H