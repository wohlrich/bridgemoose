#!/usr/bin/python
import math
import numpy
import numpy.linalg

def logisticRegression(X, y, verbose=False):
    nrows, nvars = X.shape
    if len(y) != nrows:
        raise ValueError("inputs need to have same length")

    b0 = numpy.zeros(nvars)
    itr = 0

    while itr < 100:
        itr += 1
        if verbose:
            print ("itr",itr,"beta=",b0)
        g = numpy.zeros(nvars)
        H = numpy.zeros((nvars, nvars))
        spe = 0.0        # Sum of Predicted Errors

        for i in range(nrows):
            x1 = X[i]
            y1 = y[i]
            bx = numpy.dot(b0, x1)
            sig = 1.0 / (1.0 + math.exp(-bx))
            if sig < 0.5:
                spe += sig
            else:
                spe += 1.0 - sig
            g += x1*(sig - y1)
            H += numpy.outer(x1, x1) * sig * (1.0 - sig)

        delta_b = -numpy.linalg.solve(H, g)
        err = numpy.linalg.norm(delta_b)
        if verbose:
            print(f"err is {err}")
        if err < 1e-8:
            break
        b0 += delta_b

    stderr = numpy.zeros(nvars)
    for i in range(nvars):
        x = numpy.zeros(nvars)
        x[i] = 1.0
        y = numpy.linalg.solve(H, x)
        stderr[i] = math.sqrt(y[i])

    return b0, stderr, spe/nrows
