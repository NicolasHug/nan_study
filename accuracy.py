#!/usr/bin/python3

'''script aimed to test the accuracy of 1nan (and that of 1nn) over different
boolean functions. Called by plot.py'''

import random as rd
import itertools
import sys

import tools as t

class Sn_Too_Small(Exception):
    pass

monk2 = (lambda x: sum(x) == 2) # MONK2
isEven = (lambda x: x[-1] == 1) # class is true if x is even

def main(m, n, n_exp, f=monk2):

    # construction of universe. The class of each element x is added at the end
    universe = [t.bitfield(x, m) for x in range(2**m)]
    for x in universe:
        x.append(f(x))

    if n > len(universe):
        raise Sn_Too_Small

    # average accuracy of nanMV(S, X) (= nanMV algorithm based on S evaluated
    # on all elements of X)
    avgAccNanMV = 0
    # average accuracy of nn(S, X) (= nn algorithm based on S evaluated on all
    # elements of X)
    avgAccNnS = 0
    # average accuracy of nn(AE*MV, X)
    avgAccNnAEStarMV = 0

    # theorertical accuracies: from roughest to most accurate
    # first form = acc(nn(S, X)) * lambda + acc(nn(AE*, X)) * (1 - lambda)
    avgAccNanThry1MV = 0
    # snd form = acc(nn(S, A)) * lambda + acc(nn(AE*, B)) * (1 - lambda)
    avgAccNanThry2MV = 0
    # thd form = acc(nn(S, A)) * |A| / |X| + acc(nn(AE*, B)) * |B| / |X|
    avgAccNanThry3MV = 0

    avgErr1MV = 0 # average error = difference between acc(nanMV) and formula
    avgErr2MV = 0 # average error = difference between acc(nanMV) and snd formula
    avgErr3MV = 0 # average error = difference between acc(nanMV) and trd formula

    # reminder: paper notation
    # alpha = P(1nan(x) is in S with x in X); beta = P(1nan(x) is in AE* with x
    # in X) = 1 - alpha.
    # A = {x in X | 1nan(x) is in S}
    # B = {x in X | 1nan(x) is in AE*}
    # A inter B = 0, A union B = X

    avgLambdaMV = 0 # average value of lambda = |S| / |AEMV| (a quite rough
                    # approximation of alpha)
    avgWMV = 0 # average omega = prop of correctly classified x in AE*MV
    avgWMVEst = 0 # average estimation of omega
    avgGammaMV = 0 # average gamma = |AEMV| / |X|
    avgPropAMV = 0 # average proportion (over X) of elements in A (= alpha)
    avgPropBMV = 0 # average proportion (over X) of elements in B (= beta)


    for exp in range(n_exp):
        rd.shuffle(universe) # shuffle elements of the universe
        S = universe[:n] # S = the n first elements of X
        testSet = universe[n:] # the rest is the test set

        AEMV = t.constructAEMV(S) # construct AEMV
        AEStarMV = t.getAEStar(AEMV, S) # construct AEMV* = AEMV \ S

        # compute lamda, omega and gamma
        currentLambdaMV = float(len(S)) / float(len(AEMV))
        currentWMV = t.getOmega(AEStarMV, f)
        currentWMVEst = t.getOmegaMVEst(S)
        currentGammaMV = float(len(AEMV)) / float(len(universe))

        ######################################
        ## PERFORMANCE OF NN AND NAN: START ##
        ######################################

        # perf of 1nn(S, X)
        nOKnnS = nKOnnS = 0
        # perf of 1nanMV(S, X)
        nOKnanMV= nKOnanMV= 0
        # perf of 1nn(AE*MV, X)
        nOKnnAEStarMV = nKOnnAEStarMV = 0
        # perf of 1nn(Sn, A)
        nOKnnSAMV = nKOnnSAMV = 0
        # perf of 1nn(AE*MV, B)
        nOKnnAEStarBMV = nKOnnAEStarBMV = 0

        for x in testSet:
            # 1nn(S, X)
            xnnS = t.nn(x[:-1], S, t.hamming)
            if x[-1] == xnnS[-1]:
                nOKnnS += 1
            else:
                nKOnnS += 1

            ## 1nan(S, X) ( = 1nn(AE, X))
            xnan = t.nn(x[:-1], AEMV, t.hamming)
            if x[-1] == xnan[-1]:
                nOKnanMV += 1
            else:
                nKOnanMV += 1

            # 1nn(AE*MV, X). We don't try if AE*MV is empty (which is very
            # unlikely)
            if AEStarMV:
                xnnAEStarMV = t.nn(x[:-1], AEStarMV, t.hamming)
                if x[-1] == xnnAEStarMV[-1]:
                    nOKnnAEStarMV += 1
                else:
                    nKOnnAEStarMV += 1

            else:
                nKOnnAEStarMV += 1


            # 1nn(S, A). Useful for accuracy 2 and 3
            if xnan in S: # iff x is in A
                if x[-1] == xnnS[-1]:
                    nOKnnSAMV += 1
                else:
                    nKOnnSAMV += 1

            # 1nn(S, B). Useful for accuracy 2 and 3
            else: # iff x is in B
                if x[-1] == xnnAEStarMV[-1]:
                    nOKnnAEStarBMV += 1
                else:
                    nKOnnAEStarBMV += 1

        ####################################
        ## PERFORMANCE OF NN AND NAN: END ##
        ####################################


        ######################################
        ## COMPUTATION OF CURRENT ACCURACIES #
        ######################################
        # compute accuracies
        currentAccNanMV = float(nOKnanMV) / float((nOKnanMV + nKOnanMV))
        currentAccNnS = float(nOKnnS) / float((nOKnnS + nKOnnS))
        currentAccNnAEStarMV = float(nOKnnAEStarMV) / float((nOKnnAEStarMV +
                               nKOnnAEStarMV))

        # theoretical accuracy 1
        currentAccNanThry1MV = (currentAccNnS * currentLambdaMV +
                                currentAccNnAEStarMV * (1. - currentLambdaMV))
        currentErr1MV = abs(currentAccNanMV - currentAccNanThry1MV)

        # theoretical accuracy 2
        try:
            currentAccNnSAMV = float(nOKnnSAMV) / float(nOKnnSAMV + nKOnnSAMV)
        except ZeroDivisionError:
            currentAccNnSAMV = 0
        try:
            currentAccNnAEStarBMV = float(nOKnnAEStarBMV) / float(nOKnnAEStarBMV + nKOnnAEStarBMV)
        except ZeroDivisionError:
            currentAccNnAEStarBMV = 0

        currentAccNanThry2MV = (currentAccNnSAMV * currentLambdaMV +
                                currentAccNnAEStarBMV * (1. - currentLambdaMV))
        currentErr2MV = abs(currentAccNanMV - currentAccNanThry2MV)

        # theoretical accuracy 3
        currentPropAMV = float(nOKnnSAMV + nKOnnSAMV) / len(testSet)
        currentPropBMV = float(nOKnnAEStarBMV + nKOnnAEStarBMV) / len(testSet)
        currentAccNanThry3MV = (currentAccNnSAMV * currentPropAMV +
                                currentAccNnAEStarBMV * currentPropBMV)
        currentErr3MV = abs(currentAccNanMV - currentAccNanThry3MV)


        #################################
        ## UPDATE OF AVERAGE ACCURACIES #
        #################################

        avgAccNanMV += currentAccNanMV
        avgAccNnS += currentAccNnS
        avgAccNnAEStarMV += currentAccNnAEStarMV

        avgAccNanThry1MV += currentAccNanThry1MV
        avgErr1MV += currentErr1MV
        avgAccNanThry2MV += currentAccNanThry2MV
        avgErr2MV += currentErr2MV
        avgAccNanThry3MV += currentAccNanThry3MV
        avgErr3MV += currentErr3MV

        avgLambdaMV += currentLambdaMV
        avgWMV += currentWMV
        avgWMVEst += currentWMVEst
        avgGammaMV += currentGammaMV
        avgPropAMV += currentPropAMV
        avgPropBMV += currentPropBMV

        print("m = {0:d} -- |U| = {1:d}".format(m, 2**m))
        print("n = |S| = {0:d}".format(n))
        print("|AEMV| = {0:d}".format(len(AEMV)))
        print("acc nanMV(S, X)  : {0:.3f}".format(currentAccNanMV))
        print("acc nn(S, X)     : {0:.3f}".format(currentAccNnS))
        print("-" * 10)

    # construct a dict with all the info and return it
    infos  = {
            'avgAccNanMV' : avgAccNanMV / n_exp,
            'avgAccNnS' : avgAccNnS / n_exp,
            'avgAccNnAEStarMV' : avgAccNnAEStarMV / n_exp,

            'avgAccNanThry1MV' : avgAccNanThry1MV / n_exp,
            'avgErr1MV' : avgErr1MV / n_exp,
            'avgAccNanThry2MV' : avgAccNanThry2MV / n_exp,
            'avgErr2MV' : avgErr2MV / n_exp,
            'avgAccNanThry3MV' : avgAccNanThry3MV / n_exp,
            'avgErr3MV' : avgErr3MV / n_exp,

            'avgLambdaMV' : avgLambdaMV/ n_exp,
            'avgWMV' : avgWMV / n_exp,
            'avgWMVEst' : avgWMVEst / n_exp,
            'avgGammaMV' : avgGammaMV / n_exp,
            'avgPropAMV' : avgPropAMV / n_exp,
            'avgPropBMV' : avgPropBMV /n_exp,
            }

    print("-" * 10)

    # print dict on the way
    for key, val in infos.items():
        print(key.ljust(15) , "{0:.3f}".format(val))

    return infos


if __name__ == "__main__":
    # execute only if run as a script
    if len(sys.argv) != 4:
        print("usage : python accuracy m n n_exp")
        exit(1)
    m = int(sys.argv[1])
    n = int(sys.argv[2])
    n_exp = int(sys.argv[3])
    main(m, n, n_exp)
