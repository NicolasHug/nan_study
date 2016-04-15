""" Tools needed basically everywhere """

from collections import defaultdict

def bitfield(x, m):
    """binary representation (list) of integer x over m bits"""
    b = [1 if digit=='1' else 0 for digit in bin(x)[2:]]
    for i in range(len(b), m):
        b.insert(0, 0)
    return b

def isSolvableBool(ai, bi, ci):
    """return true if boolean analogy is solvable"""
    return ai == bi or ai == ci

def isSolvableVect(a, b, c):
    """return true if ALL components are solvable"""
    return all(isSolvableBool(ai, bi, ci) for (ai, bi, ci) in zip(a, b, c))

def solveBool(ai, bi, ci):
    """return boolean solution of analogical equation. Undefined if not
    solvable"""
    if ai == bi: return ci
    else: return bi

def solveVect(a, b, c):
   """return d, the vector solution of analogical equation"""
   return [solveBool(ai, bi, ci) for (ai, bi, ci) in zip(a, b, c)]

def analogyStandsBool(ai, bi, ci, di):
    """return true if ai:bi::ci:di (boolean analogy) stands"""
    if ai == bi: return ci == di
    else: return bi == di and ci == ai

def analogyStandsVect(a, b, c, d):
    """return true if a:b::c:d (boolean vector analogy) stands"""
    return all(analogyStandsBool(*abcd) for abcd in zip(a, b, c, d))

def tripletGenerator(S):
    """Generator that generates all elements of S^3"""
    for a in S:
        for b in S:
            for c in S:
                yield (a, b, c)

def nanOldStyle(x, Sn):
    """return the estimated class of x using 1-Miclet
       Elements of Sn must have their class as last entry"""

    # return class solution of the **first** 3-tuple (a, b, c) of Sn^3 found
    # such that a:b::c:x and (a, b, c) class solvable.
    # Note that we (wrongly) assume that such a tuple exist (i.e. we consider
    # there exists one 3-tuple such that AD is null).
    # As only the first tuple is returned, it's a good idea to shuffle Sn each
    # time you call nanOldStyle, else the elected 3-tuple will often be the
    # same
    for a, b, c in tripletGenerator(Sn):
        # on regarde si a:b::c:x (sans prendre en compte les classes)
        if analogyStandsVect(a[:-1], b[:-1], c[:-1], x[:-1]):
            if isSolvableBool(a[-1], b[-1], c[-1]):
                return solveBool(a[-1], b[-1], c[-1])

def constructAEMiclet(Sn):
    """return the analogical extension set of the sample set Sn, Miclet style:
    This means that we allow memebers of AEMSn not to be in B^n but in R^n.
    Classes however stay in B."""
    sols = defaultdict(lambda: defaultdict(int))

    for a, b, c in tripletGenerator(Sn):
        if isSolvableBool(a[-1], b[-1], c[-1]):
            d = [abs(ci - ai + bi) for (ai, bi, ci) in zip(a, b, c)]
            d[-1] = solveBool(a[-1], b[-1], c[-1])
            dtuple = tuple(d[:-1])
            dclass = d[-1]
            sols[dtuple][dclass] += 1

    AEMiclet = [x for x in Sn]
    for x, vals in sols.items():
        xlist = list(x)
        if xlist in (x[:-1] for x in AEMiclet): continue
        maj_class = max(vals.keys(), key=lambda k: vals[k])
        AEMiclet.append(xlist + [maj_class])

    return AEMiclet



def constructAE(Sn):
    """return the analogical extension set of the sample set Sn
       we avoid ANY doubles, plus all elements of AE will be unique.
       In the case where an element x of AE(Sn) has two
       predicted classes, then we discard BOTH
       Elements of Sn must have their class as last entry"""

    aux = [] # will contain every solution d, with doubles
    for a, b, c in tripletGenerator(Sn):
        if isSolvableVect(a, b, c):
            d = solveVect(a, b, c)
            if d not in aux:
                aux.append(d)

    def hasADouble(x, aux):
        """return true if x is already in aux with another class"""
        for y in aux:
            if y[:-1] == x[:-1] and y[-1] != x[-1]:
                return True
        return False

    # AESn = all elements from aux that do not have doubles
    AESn = [x for x in aux if not hasADouble(x, aux)]

    # elements from Sn might have been discarded (because of a double)
    # we need to add them again
    for x in Sn:
        if x not in AESn:
            AESn.append(x)

    return AESn

def constructAEMV(Sn):
    """Return the analogical extension set of Sn where we a majority vote
    procedure is applied for calculating the analogical labels"""

    # sols is a dict with elements as keys and counts the number of 3-tuples
    # that predicted 1 or 0.
    sols = defaultdict(lambda: defaultdict(int))

    for a, b, c in tripletGenerator(Sn):
        if isSolvableVect(a, b, c):
            d = solveVect(a, b, c)
            dtuple = tuple(d[:-1])
            dclass = d[-1]
            sols[dtuple][dclass] += 1

    # Majority vote procedure
    AEMV = [x for x in Sn]
    for x, vals in sols.items():
        xlist = list(x)
        if xlist in (x[:-1] for x in AEMV): continue
        maj_class = max(vals.keys(), key=lambda k: vals[k])
        AEMV.append(xlist + [maj_class])

    return AEMV

def getOmegaMVEst(Sn):
    """Return an estimation of Omega from Sn"""

    sols = defaultdict(lambda: defaultdict(int))

    for a, b, c in tripletGenerator(Sn):
        if a is b or a is c or b is c: continue
        if isSolvableVect(a, b, c):
            d = solveVect(a, b, c)
            dtuple = tuple(d[:-1])
            dclass = d[-1]
            sols[dtuple][dclass] += 1

    nOK = nKO = 0
    for x in Sn:
        xtuple = tuple(x[:-1])
        xclass = x[-1]
        if xtuple not in sols: continue
        maj_class = max(sols[xtuple].keys(), key=lambda k: sols[xtuple][k])

        if maj_class == xclass:
            nOK += 1
        else:
            nKO += 1

    try:
        estW = nOK / (nOK + nKO)
    except ZeroDivisionError:
        estW = 0

    return estW


def getAEStar(AESn, Sn):
    """return AE* = AESn\Sn"""
    AEStar = [x for x in AESn if x not in Sn]
    return AEStar

def getOmega(AEStar, f):
    """return proportion of correctly classified elements in AEStar wrt to the
    function f"""
    nOK = sum(x[-1] == f(x[:-1]) for x in AEStar)
    try:
        w = float(nOK) / float(len(AEStar))
    except ZeroDivisionError:
        print("Warning: AE* is empty!")
        w = 0
    return w

def l1Dist(x, y):
    """return |x - y|_1    (l1 norm)"""
    return sum(abs(xi - yi) for (xi, yi) in zip(x, y))

def hamming(x, y):
    """return the hamming distance between two vectors x and y."""

    # The implem is 'symbolic', meaning that vectors can be coded not with just
    # zeros and ones, but with anything at all.
    # hamming([0, 1], [0, 1]) = 0 + 0
    # hamming([0, 1], [0, 2]) = 0 + 1
    # hamming([0, 1], [0, 5]) = 0 + 1
    # This is usefull for the monk2 dataset for ex., where some binary features
    # are coded as '1' or '2'.
    return sum(xi != yi for (xi,yi) in zip(x, y))

def nn(x, S, dist):
    """return the 1nn of x in the set S using given distance function
       x is a boolean vector (without the class at the end)
       but all elements of S have their true class at the end"""

    # note that there might be more than on minimal item. min will return the
    # first one ecountered
    return min(S, key=lambda y: dist(x, y[:-1]))
