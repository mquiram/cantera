import numpy as np
import matplotlib.pyplot as plt

def main():
    pi = np.linspace(10, 60, 1000)

    def piecewise(mask, expr, fill=np.nan):
        out = np.full_like(pi, fill, dtype=float)
        out[mask] = expr[mask]
        return out

    # Original (1981)
    orig = 40 + 2.0 * pi

    # CAEP/2
    caep2 = 32 + 1.6 * pi

    # CAEP/4
    caep4_low  = piecewise(pi <= 30, 19 + 1.6 * pi)
    caep4_high = piecewise((pi > 30) & (pi < 62.5), 7 + 2.0 * pi)

    # CAEP/6
    caep6_low  = piecewise(pi <= 30, 16.72 + 1.4080 * pi)
    caep6_high = piecewise((pi > 30) & (pi < 82.6), -1.04 + 2.0 * pi)

    # CAEP/8
    caep8_low  = piecewise(pi <= 30, 7.88 + 1.4080 * pi)
    caep8_high = piecewise(pi > 30, -9.88 + 2.0 * pi)


    plt.figure(figsize=(10, 6))
    plt.plot(pi, orig, label="CAEP: 1981", linewidth=5)
    plt.plot(pi, caep2,      label="CAEP/2: 1993", linewidth=5)
    plt.plot(pi, caep4_low,  label="CAEP/4: 1999", color='green', linewidth=5)
    plt.plot(pi, caep4_high, color='green', linewidth=5)
    plt.plot(pi, caep6_low,  label="CAEP/6: 2005", color='purple', linewidth=5)
    plt.plot(pi, caep6_high, color='purple', linewidth=5)
    plt.plot(pi, caep8_low,  label="CAEP/8: 2011", color='red', linewidth=5)
    plt.plot(pi, caep8_high, color='red', linewidth=5)


    """ for x in [30, 62.5, 82.6]:
        if 10 <= x <= 60:
            plt.axvline(x=x, linestyle="--", linewidth=1) """

    plt.xlabel("Overall Pressure Ratio, πₒₒ", fontsize=20)
    plt.ylabel(r"NO$_x$ Standard (D$_p$/Fₒₒ) [g/kN]", fontsize=20)
    plt.title(r"History of ICAO NO$_x$ Emissions Standards", fontsize=22)
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.xlim(10, 60)
    plt.grid(True, which="both", linewidth=0.8, alpha=0.5)
    plt.legend(loc="best", fontsize=18)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
