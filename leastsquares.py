import numpy as np
import random as rand
import matplotlib.pyplot as plt
import math


def errorCalc(a, y, A):
    y_found = A.dot(a)
    error = np.sqrt(np.sum(np.power((y - y_found), 2)))
    return error


def coefDet(e):
    y_mean = np.average(y)
    coef_det = 1 - (e**2) / (np.sum(np.power((y - y_mean), 2)))
    return coef_det


fReport = open("report.txt", "w")  # Make a new file in write mode
fReport.write("error:                       coefficent of determination:\n")


t, y = np.loadtxt("/Users/sam/Desktop/sampython/ame208hw/hw5/data.txt")
N = len(t)
best_coefs = np.zeros(3)
smallest_error = 10000
for w in np.linspace(2 * np.pi, 0, 1000, endpoint=False):

    # outer loop, all is does it keep w constant and tries at different values.
    # avoids w = 0, as if w = 0, sin(0) = 0 which results in an entire column being a 0 vector, which creates a singular matrix

    A = np.array([[1.0, t[i], t[i] * np.sin(w * t[i])] for i in range(N)])
    # Creates matrix A

    b = np.array([y[i] for i in range(N)])
    # Creates vector b

    AT = A.transpose()
    ATA = AT.dot(A)
    ATb = AT.dot(b)
    a = np.linalg.solve(ATA, ATb)
    current_error = errorCalc(a, y, A)
    current_coef_det = coefDet(current_error)

    # updates the best coefficents, a, if the error is lowest
    if current_error < smallest_error:
        smallest_error = current_error
        best_coefs = a

    # reporting the error
    fReport.write(
        str(current_error) + "                 " + str(current_coef_det) + "\n"
    )


fReport.close()

print(best_coefs, smallest_error)
# Plot data and best fit
a0, a1, a2 = best_coefs
print(f"Best coefficients: a0={a0:.2e}, a1={a1:.2e}, a2={a2:.2e}")
print(f"Smallest error: {smallest_error}")

# Plot data and best fit
plt.style.use("Solarize_Light2")
plt.scatter(t, y, label="Experimental Data")
plt.plot(
    t,
    A.dot(best_coefs),
    color="grey",
    linewidth=4,
    label=rf"Best Fit: $y = {a0:.2f} + ({a1:.2e}t) + ({a2:.2e}t) \sin(wt)$",
)
plt.xlabel("Time (t)")
plt.ylabel("Measured Value (y)")
plt.title(
    r"Linear Least Squares Fit of $y = a_0 + a_1 t + a_2 t \sin(w t)$ to Experimental Data"
)
plt.legend()
plt.show()
